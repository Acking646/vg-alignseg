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
from scripts.eval_utils import save_visualization, segmentation_metrics_from_prediction  # noqa: E402
from scripts.train_v2_result import cached_backbone_outputs, collate_batch, seed_everything  # noqa: E402
from scripts.train_v4_part_transfer import compose_from_actor_logits, prediction_to_logits  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    if value.strip() == "":
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V4 with fixed source views and target-only metrics.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=194)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--source-views", type=str, default="0,2,4,6")
    parser.add_argument(
        "--target-views",
        type=str,
        default="",
        help="Optional comma-separated target views. Defaults to all non-source views.",
    )
    parser.add_argument("--thresholds", type=str, default="-4,-3,-2.5,-2,-1.5,-1,-0.75,-0.5,0")
    parser.add_argument("--logit-merge", choices=("max", "mean"), default="max")
    parser.add_argument("--copy-source-views", action="store_true")
    parser.add_argument("--eval-all-views", action="store_true")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--viz-samples", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=25)
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


def fixed_source_actor_logits(
    model: VGAlignSegV4PartTransfer,
    images: torch.Tensor,
    target: torch.Tensor,
    actor_ids: Iterable[int],
    cached: dict[str, torch.Tensor],
    num_classes: int,
    source_views: list[int],
    logit_merge: str,
    copy_source_views: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    logits_by_actor = []
    class_ids = []
    for actor_id in actor_ids:
        class_id = int(actor_id) - 1
        if class_id <= 0 or class_id >= num_classes:
            continue
        target_actor = (target == class_id).float()
        source_logits = []
        visible_sources = []
        for view_idx in source_views:
            if view_idx < 0 or view_idx >= target.shape[1]:
                continue
            source_mask = target_actor[:, view_idx]
            if float(source_mask.sum().item()) <= 0.0:
                continue
            source_view = torch.tensor([view_idx], device=target.device, dtype=torch.long)
            outputs = model.forward_from_backbone_outputs(
                images=images,
                backbone_outputs=cached,
                source_mask=source_mask,
                source_view=source_view,
                output_size=images.shape[-2:],
            )
            source_logits.append(outputs["binary_logits"][:, :, 0])
            visible_sources.append((source_mask, source_view))
        if not source_logits:
            continue

        stacked = torch.stack(source_logits, dim=0)
        actor_logits = stacked.mean(dim=0) if logit_merge == "mean" else stacked.max(dim=0).values
        if copy_source_views:
            for source_mask, source_view in visible_sources:
                view_idx = int(source_view.item())
                actor_logits[:, view_idx] = torch.where(source_mask > 0.5, 30.0, -30.0)
        logits_by_actor.append(actor_logits)
        class_ids.append(class_id)

    if not logits_by_actor:
        return None, torch.empty(0, device=target.device, dtype=target.dtype)
    return torch.stack(logits_by_actor, dim=1), torch.tensor(class_ids, device=target.device, dtype=target.dtype)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    source_views = parse_csv_ints(args.source_views)
    target_views = parse_csv_ints(args.target_views)
    if not args.eval_all_views and not target_views:
        target_views = [view_idx for view_idx in range(args.views) if view_idx not in set(source_views)]
    if args.eval_all_views:
        target_views = list(range(args.views))
    thresholds = parse_csv_floats(args.thresholds)

    model, train_args, meta = load_model(args.checkpoint, args.device)
    num_classes = int(meta.get("num_classes", train_args.get("num_classes", 117)))
    dataset = ResultMultiViewDataset(
        args.data_root,
        split=args.split,
        views=args.views,
        image_size=args.image_size,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_batch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    totals_by_threshold = {
        threshold: {"pixel_errors": 0, "pixel_count": 0, "mean_iou_sum": 0.0, "batches": 0}
        for threshold in thresholds
    }
    viz_records = []
    per_object = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_objects is not None and batch_idx >= args.max_objects:
                break
            if args.log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % args.log_every == 0):
                print(f"eval progress: object {batch_idx + 1}/{len(loader)}", flush=True)
            images = batch["images"].to(args.device, non_blocking=True)
            target = batch["masks"].to(device=args.device, dtype=torch.long, non_blocking=True)
            cached = cached_backbone_outputs(model, images, batch["object_id"], args.cache_dir, args.device)
            actor_logits, class_ids = fixed_source_actor_logits(
                model=model,
                images=images,
                target=target,
                actor_ids=batch["actor_ids"][0],
                cached=cached,
                num_classes=num_classes,
                source_views=source_views,
                logit_merge=args.logit_merge,
                copy_source_views=args.copy_source_views,
            )
            preds_by_threshold = {}
            object_scores = {}
            for threshold in thresholds:
                pred = compose_from_actor_logits(actor_logits, class_ids, target, threshold)
                pred_eval = pred[:, target_views]
                target_eval = target[:, target_views]
                metrics = segmentation_metrics_from_prediction(pred_eval, target_eval, num_classes)
                totals = totals_by_threshold[threshold]
                totals["pixel_errors"] += int(metrics["pixel_errors"])
                totals["pixel_count"] += int(target_eval.numel())
                totals["mean_iou_sum"] += float(metrics["mean_iou"])
                totals["batches"] += 1
                preds_by_threshold[threshold] = pred.detach().cpu().to(torch.uint8)
                object_scores[str(threshold)] = {
                    "mean_iou": metrics["mean_iou"],
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "pixel_errors": metrics["pixel_errors"],
                }
            per_object.append({"index": batch_idx, "object_id": batch["object_id"][0], "scores": object_scores})
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

    best_threshold = thresholds[0]
    best_metrics = None
    threshold_metrics = {}
    for threshold, totals in totals_by_threshold.items():
        metrics = {
            "eval_pixel_accuracy": 1.0 - totals["pixel_errors"] / max(1, totals["pixel_count"]),
            "eval_mean_iou": totals["mean_iou_sum"] / max(1, totals["batches"]),
            "eval_pixel_errors": totals["pixel_errors"],
            "eval_batches": totals["batches"],
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
        "source_views": source_views,
        "target_views": target_views,
        "copy_source_views": args.copy_source_views,
        "logit_merge": args.logit_merge,
        "best_threshold": best_threshold,
        **best_metrics,
        "threshold_metrics": threshold_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "per_object.json").write_text(json.dumps(per_object, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
