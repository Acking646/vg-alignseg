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
from models import VGAlignSegV3, VGAlignSegV4PartTransfer  # noqa: E402
from scripts.eval_utils import mask_logits_to_actor_ids, save_visualization, segmentation_metrics_from_prediction  # noqa: E402
from scripts.train_v2_result import cached_backbone_outputs, collate_batch, seed_everything  # noqa: E402
from scripts.train_v3_result import load_compatible_state  # noqa: E402
from scripts.train_v4_part_transfer import compose_from_actor_logits, prediction_to_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate V3 self-prompted V4 part transfer without GT test prompts.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "v3_pseudo_v4_eval")
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=194)
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--v3-hidden-dim", type=int, default=128)
    parser.add_argument("--v4-hidden-dim", type=int, default=128)
    parser.add_argument("--refinement-iters", type=int, default=2)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--source-mask", type=str, choices=("hard", "prob"), default="hard")
    parser.add_argument("--source-min-area", type=float, default=8.0)
    parser.add_argument("--merge", type=str, choices=("max", "mean"), default="max")
    parser.add_argument("--copy-source-views", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="-4,-3.5,-3,-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,0",
    )
    parser.add_argument("--viz-samples", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def load_v4_state(model: VGAlignSegV4PartTransfer, checkpoint_path: Path, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(source, strict=False)
    missing = [key for key in missing if not key.startswith("backbone.")]
    print(f"Loaded V4 transfer: missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def parse_thresholds(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    return values or [0.0]


def select_pseudo_sources(
    probs: torch.Tensor,
    v3_pred: torch.Tensor,
    class_id: int,
    topk: int,
    mask_mode: str,
    min_area: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    class_probs = probs[:, :, class_id]
    hard_masks = (v3_pred == class_id).float()
    scores = class_probs.sum(dim=(-2, -1))[0]
    hard_area = hard_masks.sum(dim=(-2, -1))[0]
    order = torch.argsort(scores, descending=True)

    selected = []
    for view_idx in order:
        if float(scores[int(view_idx)].item()) <= 0.0:
            continue
        if float(hard_area[int(view_idx)].item()) < min_area and len(selected) > 0:
            continue
        if mask_mode == "prob":
            source_mask = class_probs[:, int(view_idx)]
        else:
            source_mask = hard_masks[:, int(view_idx)]
            if float(source_mask.sum().item()) < min_area:
                source_mask = (class_probs[:, int(view_idx)] > 0.25).float()
        if float(source_mask.sum().item()) <= 0.0:
            continue
        selected.append((source_mask, view_idx.view(1).to(dtype=torch.long)))
        if len(selected) >= topk:
            break

    if not selected:
        view_idx = order[0].view(1).to(dtype=torch.long)
        selected.append((class_probs[:, int(view_idx.item())], view_idx))
    return selected


def actor_logits_from_pseudo_sources(
    v4_model: VGAlignSegV4PartTransfer,
    images: torch.Tensor,
    cached: dict[str, torch.Tensor],
    probs: torch.Tensor,
    v3_pred: torch.Tensor,
    actor_ids: Iterable[int],
    num_classes: int,
    topk: int,
    mask_mode: str,
    min_area: float,
    merge: str,
    copy_source_views: bool,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    logits_by_actor = []
    class_ids = []
    for actor_id in actor_ids:
        class_id = int(actor_id) - 1
        if class_id <= 0 or class_id >= num_classes:
            continue
        sources = select_pseudo_sources(
            probs=probs,
            v3_pred=v3_pred,
            class_id=class_id,
            topk=topk,
            mask_mode=mask_mode,
            min_area=min_area,
        )
        source_logits = []
        for source_mask, source_view in sources:
            outputs = v4_model.forward_from_backbone_outputs(
                images=images,
                backbone_outputs=cached,
                source_mask=source_mask,
                source_view=source_view.to(device=images.device),
                output_size=images.shape[-2:],
            )
            source_logits.append(outputs["binary_logits"][:, :, 0])

        stacked = torch.stack(source_logits, dim=0)
        if merge == "mean":
            actor_logits = stacked.mean(dim=0)
        else:
            actor_logits = stacked.max(dim=0).values
        if copy_source_views:
            for source_mask, source_view in sources:
                view_idx = int(source_view.item())
                actor_logits[:, view_idx] = torch.where(source_mask > 0.5, 30.0, -30.0)
        logits_by_actor.append(actor_logits)
        class_ids.append(class_id)

    if not logits_by_actor:
        return None, torch.empty(0, device=images.device, dtype=torch.long)
    return torch.stack(logits_by_actor, dim=1), torch.tensor(class_ids, device=images.device, dtype=torch.long)


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ResultMultiViewDataset(
        args.data_root,
        split=args.split,
        views=args.views,
        image_size=args.image_size,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for cached object-level evaluation.")

    v3_model = VGAlignSegV3(
        num_classes=dataset.num_classes,
        refine_hidden_dim=args.v3_hidden_dim,
        use_prototypes=True,
    ).to(args.device)
    load_compatible_state(v3_model, args.v3_checkpoint, args.device)
    v3_model.use_object_class_prior = True
    v3_model.eval()
    v3_model.backbone.eval()

    v4_model = VGAlignSegV4PartTransfer(
        hidden_dim=args.v4_hidden_dim,
        refinement_iters=args.refinement_iters,
    ).to(args.device)
    load_v4_state(v4_model, args.v4_checkpoint, args.device)
    v4_model.eval()
    v4_model.backbone.eval()

    thresholds = parse_thresholds(args.thresholds)
    totals_by_threshold = {
        threshold: {"pixel_errors": 0, "pixel_count": 0, "mean_iou_sum": 0.0, "batches": 0}
        for threshold in thresholds
    }
    viz_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_objects is not None and batch_idx >= args.max_objects:
                break
            if args.log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % args.log_every == 0):
                print(f"eval progress: object {batch_idx + 1}/{len(loader)}", flush=True)
            images = batch["images"].to(args.device, non_blocking=True)
            target = batch["masks"].to(device=args.device, dtype=torch.long, non_blocking=True)
            cached = cached_backbone_outputs(v3_model, images, batch["object_id"], args.cache_dir, args.device)

            v3_outputs = v3_model.forward_from_backbone_outputs(images, cached, output_size=images.shape[-2:])
            v3_logits = mask_logits_to_actor_ids(v3_outputs["final_logits"], batch["actor_ids"])
            probs = v3_logits.softmax(dim=2)
            v3_pred = probs.argmax(dim=2)

            actor_logits, class_ids = actor_logits_from_pseudo_sources(
                v4_model=v4_model,
                images=images,
                cached=cached,
                probs=probs,
                v3_pred=v3_pred,
                actor_ids=batch["actor_ids"][0],
                num_classes=dataset.num_classes,
                topk=args.topk,
                mask_mode=args.source_mask,
                min_area=args.source_min_area,
                merge=args.merge,
                copy_source_views=args.copy_source_views,
            )

            viz_preds = {} if batch_idx < args.viz_samples else None
            for threshold in thresholds:
                pred = compose_from_actor_logits(actor_logits, class_ids, target, threshold)
                metrics = segmentation_metrics_from_prediction(pred, target, dataset.num_classes)
                totals = totals_by_threshold[threshold]
                totals["pixel_errors"] += int(metrics["pixel_errors"])
                totals["pixel_count"] += int(target.numel())
                totals["mean_iou_sum"] += float(metrics["mean_iou"])
                totals["batches"] += 1
                if viz_preds is not None:
                    viz_preds[threshold] = pred.detach().cpu().to(torch.uint8)

            if viz_preds is not None:
                viz_records.append(
                    {
                        "batch_idx": batch_idx,
                        "object_id": batch["object_id"][0],
                        "view_names": batch["view_names"][0],
                        "images": images.detach().cpu(),
                        "target": target.detach().cpu(),
                        "preds": viz_preds,
                    }
                )

    best_threshold = thresholds[0]
    best_metrics = None
    threshold_metrics = {}
    for threshold, totals in totals_by_threshold.items():
        if totals["batches"] == 0:
            metrics = {"eval_pixel_accuracy": 0.0, "eval_mean_iou": 0.0, "eval_pixel_errors": 0, "eval_batches": 0}
        else:
            metrics = {
                "eval_pixel_accuracy": 1.0 - totals["pixel_errors"] / max(1, totals["pixel_count"]),
                "eval_mean_iou": totals["mean_iou_sum"] / totals["batches"],
                "eval_pixel_errors": totals["pixel_errors"],
                "eval_batches": totals["batches"],
            }
        threshold_metrics[str(threshold)] = metrics
        if best_metrics is None or metrics["eval_mean_iou"] > best_metrics["eval_mean_iou"]:
            best_threshold = threshold
            best_metrics = metrics

    assert best_metrics is not None
    viz_dir = args.output_dir / "test_viz_pseudo_prompt"
    for record in viz_records:
        pred = record["preds"][best_threshold].to(dtype=torch.long)
        logits = prediction_to_logits(pred, dataset.num_classes)
        save_visualization(
            viz_dir / f"{record['batch_idx']:04d}_{record['object_id']}.png",
            record["images"],
            record["target"],
            logits,
            record["view_names"],
        )

    output = {
        "phase": args.split,
        **best_metrics,
        "eval_best_threshold": best_threshold,
        "eval_threshold_metrics": threshold_metrics,
        "args": vars(args),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    with (args.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, **output}, default=str) + "\n")
    (viz_dir / "metadata.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(
        f"{args.split}@pseudo_v4: miou={best_metrics['eval_mean_iou']:.4f} "
        f"thr={best_threshold:.3f} acc={best_metrics['eval_pixel_accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
