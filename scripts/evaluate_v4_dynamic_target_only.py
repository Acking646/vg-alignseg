from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset  # noqa: E402
from models import VGAlignSegV4PartTransfer  # noqa: E402
from scripts.eval_utils import save_visualization  # noqa: E402
from scripts.train_v2_result import cached_backbone_outputs, collate_batch, seed_everything  # noqa: E402
from scripts.train_v4_part_transfer import choose_eval_sources, compose_from_actor_logits, prediction_to_logits  # noqa: E402


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def optional_count(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4 top-k source transfer on non-source target views only.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--source-topk", type=int, default=4)
    parser.add_argument("--thresholds", type=str, default="-4,-3.5,-3,-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,0")
    parser.add_argument("--logit-merge", choices=("max", "mean"), default="max")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--viz-samples", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[VGAlignSegV4PartTransfer, dict, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    meta = checkpoint.get("meta", {})
    model = VGAlignSegV4PartTransfer(
        checkpoint_path=train_args.get("checkpoint_path"),
        hidden_dim=int(train_args.get("hidden_dim", 128)),
        refinement_iters=int(train_args.get("refinement_iters", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model, train_args, meta


def actor_logits_with_sources(
    model: VGAlignSegV4PartTransfer,
    images: torch.Tensor,
    target: torch.Tensor,
    actor_ids: Iterable[int],
    cached: dict[str, torch.Tensor],
    num_classes: int,
    source_topk: int,
    logit_merge: str,
) -> tuple[torch.Tensor | None, torch.Tensor, dict[int, list[int]]]:
    logits_by_actor = []
    class_ids = []
    sources_by_class: dict[int, list[int]] = {}
    for actor_id in actor_ids:
        class_id = int(actor_id) - 1
        if class_id <= 0 or class_id >= num_classes:
            continue
        target_actor = (target == class_id).float()
        if float(target_actor.sum().item()) <= 0.0:
            continue
        source_logits = []
        sources = choose_eval_sources(target_actor, source_topk=source_topk)
        sources_by_class[class_id] = [int(source_view.item()) for _, source_view in sources]
        for source_mask, source_view in sources:
            outputs = model.forward_from_backbone_outputs(
                images=images,
                backbone_outputs=cached,
                source_mask=source_mask,
                source_view=source_view.to(device=target.device, dtype=torch.long),
                output_size=images.shape[-2:],
            )
            source_logits.append(outputs["binary_logits"][:, :, 0])
        stacked = torch.stack(source_logits, dim=0)
        actor_logits = stacked.mean(dim=0) if logit_merge == "mean" else stacked.max(dim=0).values
        logits_by_actor.append(actor_logits)
        class_ids.append(class_id)

    if not logits_by_actor:
        return None, torch.empty(0, device=target.device, dtype=target.dtype), sources_by_class
    return torch.stack(logits_by_actor, dim=1), torch.tensor(class_ids, device=target.device, dtype=target.dtype), sources_by_class


def target_only_actor_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    class_ids: torch.Tensor,
    sources_by_class: dict[int, list[int]],
    n_views: int,
) -> dict[str, object]:
    ious = []
    pixel_errors = 0
    pixel_count = 0
    per_class = {}
    for class_id_tensor in class_ids.detach().cpu().tolist():
        class_id = int(class_id_tensor)
        source_set = set(sources_by_class.get(class_id, []))
        target_views = [view_idx for view_idx in range(n_views) if view_idx not in source_set]
        if not target_views:
            continue
        pred_c = pred[:, target_views] == class_id
        target_c = target[:, target_views] == class_id
        union = (pred_c | target_c).sum()
        inter = (pred_c & target_c).sum()
        if int(union.item()) <= 0:
            continue
        iou = float((inter.float() / union.float()).item())
        ious.append(iou)
        errors = int((pred_c != target_c).sum().item())
        count = int(pred_c.numel())
        pixel_errors += errors
        pixel_count += count
        per_class[str(class_id)] = {
            "iou": iou,
            "source_views": sorted(source_set),
            "target_views": target_views,
            "pixel_errors": errors,
            "pixel_count": count,
        }
    return {
        "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
        "pixel_accuracy": 1.0 - pixel_errors / max(1, pixel_count),
        "pixel_errors": pixel_errors,
        "pixel_count": pixel_count,
        "actors": len(ious),
        "per_class": per_class,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    thresholds = parse_floats(args.thresholds)
    model, train_args, meta = load_model(args.checkpoint, args.device)
    num_classes = int(meta.get("num_classes", train_args.get("num_classes", 117)))
    dataset = ResultMultiViewDataset(
        args.data_root,
        split=args.split,
        views=args.views,
        image_size=args.image_size,
        train_count=optional_count(args.train_count),
        val_count=optional_count(args.val_count),
        test_count=optional_count(args.test_count),
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_batch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals = {
        threshold: {"mean_iou_sum": 0.0, "pixel_errors": 0, "pixel_count": 0, "actors": 0, "objects": 0}
        for threshold in thresholds
    }
    per_object = []
    viz_records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_objects is not None and batch_idx >= args.max_objects:
                break
            if args.log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % args.log_every == 0):
                print(f"eval progress: object {batch_idx + 1}/{len(loader)}", flush=True)
            images = batch["images"].to(args.device, non_blocking=True)
            target = batch["masks"].to(device=args.device, dtype=torch.long, non_blocking=True)
            cached = cached_backbone_outputs(model, images, batch["object_id"], args.cache_dir, args.device)
            actor_logits, class_ids, sources_by_class = actor_logits_with_sources(
                model=model,
                images=images,
                target=target,
                actor_ids=batch["actor_ids"][0],
                cached=cached,
                num_classes=num_classes,
                source_topk=args.source_topk,
                logit_merge=args.logit_merge,
            )
            object_scores = {}
            preds_by_threshold = {}
            for threshold in thresholds:
                pred = compose_from_actor_logits(actor_logits, class_ids, target, threshold)
                metrics = target_only_actor_metrics(pred, target, class_ids, sources_by_class, args.views)
                total = totals[threshold]
                total["mean_iou_sum"] += float(metrics["mean_iou"])
                total["pixel_errors"] += int(metrics["pixel_errors"])
                total["pixel_count"] += int(metrics["pixel_count"])
                total["actors"] += int(metrics["actors"])
                total["objects"] += 1
                object_scores[str(threshold)] = {
                    "mean_iou": metrics["mean_iou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "pixel_errors": metrics["pixel_errors"],
                    "actors": metrics["actors"],
                }
                preds_by_threshold[threshold] = pred.detach().cpu().to(torch.uint8)
            per_object.append(
                {
                    "index": batch_idx,
                    "object_id": batch["object_id"][0],
                    "scores": object_scores,
                    "sources_by_class": {str(k): v for k, v in sources_by_class.items()},
                }
            )
            if batch_idx < args.viz_samples:
                viz_records.append(
                    {
                        "batch_idx": batch_idx,
                        "object_id": batch["object_id"][0],
                        "view_names": batch["view_names"][0],
                        "images": images.detach().cpu(),
                        "target": target.detach().cpu(),
                        "preds_by_threshold": preds_by_threshold,
                    }
                )

    threshold_metrics = {}
    best_threshold = thresholds[0]
    best_metrics = None
    for threshold, total in totals.items():
        metrics = {
            "eval_mean_iou": total["mean_iou_sum"] / max(1, total["objects"]),
            "eval_pixel_accuracy": 1.0 - total["pixel_errors"] / max(1, total["pixel_count"]),
            "eval_pixel_errors": total["pixel_errors"],
            "eval_pixel_count": total["pixel_count"],
            "eval_actor_instances": total["actors"],
            "eval_batches": total["objects"],
        }
        threshold_metrics[str(threshold)] = metrics
        if best_metrics is None or metrics["eval_mean_iou"] > best_metrics["eval_mean_iou"]:
            best_threshold = threshold
            best_metrics = metrics

    assert best_metrics is not None
    viz_dir = args.output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    for record in viz_records:
        pred = record["preds_by_threshold"][best_threshold].to(dtype=torch.long)
        logits = prediction_to_logits(pred, num_classes)
        save_visualization(
            viz_dir / f"{record['batch_idx']:04d}_{record['object_id']}.png",
            record["images"],
            record["target"],
            logits,
            record["view_names"],
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "shuffle_split": args.shuffle_split,
        "split_seed": args.split_seed,
        "train_count": args.train_count,
        "test_count": len(dataset),
        "source_topk": args.source_topk,
        "logit_merge": args.logit_merge,
        "metric": "actor-wise IoU on non-source target views only",
        "best_threshold": best_threshold,
        **best_metrics,
        "threshold_metrics": threshold_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "per_object.json").write_text(json.dumps(per_object, indent=2), encoding="utf-8")
    dataset.write_manifest(args.output_dir / f"{args.split}_manifest.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
