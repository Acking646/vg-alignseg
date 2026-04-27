from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset  # noqa: E402
from models import VGAlignSegV4PartTransfer  # noqa: E402
from scripts.eval_utils import save_visualization, segmentation_metrics_from_prediction  # noqa: E402
from scripts.train_v2_result import (  # noqa: E402
    barrier,
    cached_backbone_outputs,
    cleanup_distributed,
    collate_batch,
    is_rank0,
    move_optimizer_state_to_device,
    print_rank0,
    reduce_mean,
    reduce_sum,
    save_checkpoint,
    seed_everything,
    setup_distributed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train prompt-conditioned VG-AlignSeg V4 part transfer.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "v4_part_transfer")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=194)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--eval-split", type=str, choices=("train", "val", "test"), default="test")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--eval-max-objects", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--refinement-iters", type=int, default=2)
    parser.add_argument("--train-source-policy", type=str, choices=("random", "largest"), default="random")
    parser.add_argument(
        "--train-source-topk",
        type=int,
        default=1,
        help="Use the top-k selected source views per actor during training. Use <=0 for all visible views.",
    )
    parser.add_argument(
        "--train-logit-merge",
        type=str,
        choices=("max", "mean"),
        default="max",
        help="How to merge logits from multiple training source prompts for one actor.",
    )
    parser.add_argument(
        "--train-copy-source-views",
        action="store_true",
        help="Inject the prompted source masks into merged training logits before computing the loss.",
    )
    parser.add_argument(
        "--train-independent-sources",
        action="store_true",
        help="Train each selected source prompt independently to avoid retaining multiple source graphs.",
    )
    parser.add_argument(
        "--train-exclude-source-views-from-loss",
        action="store_true",
        help="Do not compute transfer loss on the prompted source views.",
    )
    parser.add_argument(
        "--train-fixed-source-views",
        type=str,
        default=None,
        help="Comma-separated fixed source view ids for protocol-specific fine-tuning, e.g. '0,3,5,6'.",
    )
    parser.add_argument(
        "--train-loss-views",
        type=str,
        default=None,
        help="Comma-separated view ids on which to compute the binary transfer loss.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument("--bce-loss-weight", type=float, default=1.0)
    parser.add_argument("--focal-loss-weight", type=float, default=20.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--tversky-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.2)
    parser.add_argument("--boundary-head-loss-weight", type=float, default=0.05)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--compose-threshold", type=float, default=0.0)
    parser.add_argument(
        "--eval-thresholds",
        type=str,
        default=None,
        help="Comma-separated logit thresholds to sweep during eval, e.g. '-1,0,0.5,1'.",
    )
    parser.add_argument(
        "--eval-source-topk",
        type=int,
        default=1,
        help="Use the top-k visible source views per actor during eval. Use <=0 for all visible views.",
    )
    parser.add_argument(
        "--eval-logit-merge",
        type=str,
        choices=("max", "mean"),
        default="max",
        help="How to merge logits from multiple source-view prompts for one actor.",
    )
    parser.add_argument(
        "--eval-copy-source-views",
        action="store_true",
        help="Inject the prompted source masks as exact logits for the source views during eval composition.",
    )
    parser.add_argument("--target-eval-miou", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-log-every", type=int, default=25)
    parser.add_argument("--viz-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-viz-samples", type=int, default=4)
    return parser.parse_args()


def load_trainable_state(model: VGAlignSegV4PartTransfer, checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(source, strict=False)
    missing = [key for key in missing if not key.startswith("backbone.")]
    print(f"Loaded V4 checkpoint with missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def binary_focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, prob, 1.0 - prob).clamp(1e-6, 1.0)
    alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1.0 - alpha))
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def binary_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def binary_tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.3, beta: float = 0.7) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - prob) * target).sum(dim=dims)
    return 1.0 - ((tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)).mean()


def soft_boundary(x: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def binary_boundary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    pred_b = soft_boundary(prob.reshape(-1, 1, *prob.shape[-2:]))
    target_b = soft_boundary(target.reshape(-1, 1, *target.shape[-2:]))
    inter = (pred_b * target_b).sum(dim=(1, 2, 3))
    denom = pred_b.sum(dim=(1, 2, 3)) + target_b.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0)).mean()


def binary_boundary_target(target: torch.Tensor) -> torch.Tensor:
    return soft_boundary(target.reshape(-1, 1, *target.shape[-2:])).reshape_as(target)


def binary_metrics(logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred = logits > 0.0
    gt = target > 0.5
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    errors = int((pred != gt).sum().item())
    return {
        "binary_iou": float((inter / union.clamp_min(1.0)).item()),
        "binary_accuracy": float((pred == gt).float().mean().item()),
        "binary_errors": errors,
    }


def parse_thresholds(args: argparse.Namespace) -> list[float]:
    if args.eval_thresholds is None:
        return [float(args.compose_threshold)]
    values = [float(item.strip()) for item in args.eval_thresholds.split(",") if item.strip()]
    return values or [float(args.compose_threshold)]


def parse_view_ids(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def optional_count(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def select_loss_views(
    total_views: int,
    configured_views: list[int],
    source_views: list[int],
    exclude_source_views: bool,
) -> list[int]:
    views = configured_views if configured_views else list(range(total_views))
    if exclude_source_views:
        source_set = set(source_views)
        views = [view_idx for view_idx in views if view_idx not in source_set]
    return [view_idx for view_idx in views if 0 <= view_idx < total_views]


def choose_source_views(
    target: torch.Tensor,
    source_policy: str = "random",
    source_topk: int = 1,
) -> list[torch.Tensor]:
    view_area = target.sum(dim=(-2, -1))[0]
    visible = torch.nonzero(view_area > 0, as_tuple=False).flatten()
    if visible.numel() == 0:
        k = max(1, source_topk) if source_topk > 0 else 1
        return [torch.tensor([0], device=target.device, dtype=torch.long) for _ in range(k)]

    if source_topk <= 0:
        requested_k = int(visible.numel())
    else:
        requested_k = max(1, source_topk)
    sample_k = min(requested_k, int(visible.numel()))

    if source_policy == "largest":
        order = torch.argsort(view_area[visible], descending=True)
        chosen = visible[order[:sample_k]]
    else:
        weights = view_area[visible].float().clamp_min(1.0).sqrt().cpu()
        sample = torch.multinomial(weights, sample_k, replacement=False).to(device=visible.device)
        chosen = visible[sample]

    if source_topk > 0 and int(chosen.numel()) < requested_k:
        repeat = chosen[-1:].expand(requested_k - int(chosen.numel()))
        chosen = torch.cat([chosen, repeat], dim=0)
    return [view_idx.view(1).to(dtype=torch.long) for view_idx in chosen]


def sources_from_views(target: torch.Tensor, source_views: Iterable[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    sources = []
    for source_view in source_views:
        source_view = source_view.to(device=target.device, dtype=torch.long)
        source_mask = target[:, int(source_view.item())]
        sources.append((source_mask, source_view))
    return sources


def choose_training_prompt(
    batch: Dict[str, object],
    device: str,
    source_policy: str = "random",
    source_topk: int = 1,
    fixed_source_views: list[int] | None = None,
) -> Dict[str, object]:
    masks = batch["masks"].to(device=device, dtype=torch.long)
    actor_ids = list(batch["actor_ids"][0])
    if not actor_ids:
        raise ValueError("Object has no actors.")

    candidates = []
    areas = []
    for actor_id in actor_ids:
        class_id = int(actor_id) - 1
        actor_mask = masks == class_id
        if fixed_source_views:
            visible = any(float(actor_mask[:, view_idx].sum().item()) > 0.0 for view_idx in fixed_source_views)
            if not visible:
                continue
        candidates.append(actor_id)
        areas.append(actor_mask.sum().float().clamp_min(1.0))
    if not candidates:
        candidates = actor_ids
        areas = [(masks == (int(actor_id) - 1)).sum().float().clamp_min(1.0) for actor_id in candidates]
    weights = torch.stack([area.rsqrt() for area in areas])
    actor_idx = int(torch.multinomial(weights.cpu(), 1).item())
    actor_id = int(candidates[actor_idx])
    class_id = actor_id - 1
    target = (masks == class_id).float()

    if fixed_source_views:
        visible_pairs = [
            (view_idx, target[:, view_idx].sum().float().clamp_min(1.0))
            for view_idx in fixed_source_views
            if 0 <= view_idx < target.shape[1] and float(target[:, view_idx].sum().item()) > 0.0
        ]
        if source_topk > 0 and visible_pairs:
            sample_k = max(1, min(source_topk, len(visible_pairs)))
            if source_policy == "largest":
                visible_pairs = sorted(visible_pairs, key=lambda item: float(item[1].item()), reverse=True)[:sample_k]
            elif len(visible_pairs) > sample_k:
                weights = torch.stack([area.sqrt() for _, area in visible_pairs]).cpu()
                chosen_idx = torch.multinomial(weights, sample_k, replacement=False).tolist()
                visible_pairs = [visible_pairs[idx] for idx in chosen_idx]
        visible_fixed = [torch.tensor([view_idx], device=target.device, dtype=torch.long) for view_idx, _ in visible_pairs]
        source_views = visible_fixed or choose_source_views(target, source_policy=source_policy, source_topk=source_topk)
    else:
        source_views = choose_source_views(target, source_policy=source_policy, source_topk=source_topk)
    sources = sources_from_views(target, source_views)
    source_mask, source_view = sources[0]
    return {
        "actor_id": actor_id,
        "class_id": class_id,
        "target": target,
        "source_mask": source_mask,
        "source_view": source_view.to(device=masks.device, dtype=torch.long),
        "sources": sources,
        "source_views": [int(view.item()) for _, view in sources],
    }


def choose_eval_sources(target_actor: torch.Tensor, source_topk: int = 1) -> list[tuple[torch.Tensor, torch.Tensor]]:
    view_area = target_actor.sum(dim=(-2, -1))[0]
    visible = torch.nonzero(view_area > 0, as_tuple=False).flatten()
    if visible.numel() == 0:
        source_view = torch.tensor([0], device=target_actor.device, dtype=torch.long)
        return [(target_actor[:, 0], source_view)]

    order = torch.argsort(view_area[visible], descending=True)
    visible = visible[order]
    if source_topk > 0:
        visible = visible[: max(1, min(source_topk, int(visible.numel())))]

    sources = []
    for view_idx in visible:
        source_view = view_idx.view(1).to(dtype=torch.long)
        source_mask = target_actor[:, int(view_idx.item())]
        sources.append((source_mask, source_view))
    return sources


def actor_logits_for_object(
    model: VGAlignSegV4PartTransfer,
    images: torch.Tensor,
    target: torch.Tensor,
    actor_ids: Iterable[int],
    cached: Dict[str, torch.Tensor],
    num_classes: int,
    source_topk: int = 1,
    logit_merge: str = "max",
    copy_source_views: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    logits_by_actor = []
    class_ids = []
    for actor_id in actor_ids:
        class_id = int(actor_id) - 1
        if class_id <= 0 or class_id >= num_classes:
            continue
        target_actor = (target == class_id).float()
        if float(target_actor.sum().item()) <= 0.0:
            continue
        source_logits = []
        sources = choose_eval_sources(target_actor, source_topk=source_topk)
        for source_mask, source_view in sources:
            outputs = model.forward_from_backbone_outputs(
                images=images,
                backbone_outputs=cached,
                source_mask=source_mask,
                source_view=source_view,
                output_size=images.shape[-2:],
            )
            source_logits.append(outputs["binary_logits"][:, :, 0])

        stacked_logits = torch.stack(source_logits, dim=0)
        if logit_merge == "mean":
            actor_logits = stacked_logits.mean(dim=0)
        else:
            actor_logits = stacked_logits.max(dim=0).values

        if copy_source_views:
            for source_mask, source_view in sources:
                view_idx = int(source_view.item())
                actor_logits[:, view_idx] = torch.where(source_mask > 0.5, 30.0, -30.0)

        logits_by_actor.append(actor_logits)
        class_ids.append(class_id)

    if not logits_by_actor:
        return None, torch.empty(0, device=target.device, dtype=target.dtype)
    return torch.stack(logits_by_actor, dim=1), torch.tensor(class_ids, device=target.device, dtype=target.dtype)


def merge_source_outputs(
    outputs_by_source: list[Dict[str, torch.Tensor]],
    sources: list[tuple[torch.Tensor, torch.Tensor]],
    logit_merge: str = "max",
    copy_source_views: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_stack = torch.stack([outputs["binary_logits"][:, :, 0] for outputs in outputs_by_source], dim=0)
    boundary_stack = torch.stack([outputs["boundary_logits"][:, :, 0] for outputs in outputs_by_source], dim=0)
    if logit_merge == "mean":
        logits = logits_stack.mean(dim=0)
        boundary_logits = boundary_stack.mean(dim=0)
    else:
        logits = logits_stack.max(dim=0).values
        boundary_logits = boundary_stack.max(dim=0).values

    if copy_source_views:
        for source_mask, source_view in sources:
            view_idx = int(source_view.item())
            logits[:, view_idx] = torch.where(source_mask > 0.5, 30.0, -30.0)
    return logits, boundary_logits


def compose_from_actor_logits(
    actor_logits: torch.Tensor | None,
    class_ids: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    pred = torch.zeros_like(target)
    if actor_logits is None or class_ids.numel() == 0:
        return pred
    best_logit, best_idx = actor_logits.max(dim=1)
    chosen = class_ids[best_idx]
    return torch.where(best_logit > threshold, chosen, pred)


def compose_prediction_for_object(
    model: VGAlignSegV4PartTransfer,
    images: torch.Tensor,
    target: torch.Tensor,
    actor_ids: Iterable[int],
    cached: Dict[str, torch.Tensor],
    num_classes: int,
    threshold: float,
    source_topk: int = 1,
    logit_merge: str = "max",
    copy_source_views: bool = False,
) -> torch.Tensor:
    actor_logits, class_ids = actor_logits_for_object(
        model,
        images,
        target,
        actor_ids,
        cached,
        num_classes,
        source_topk=source_topk,
        logit_merge=logit_merge,
        copy_source_views=copy_source_views,
    )
    return compose_from_actor_logits(actor_logits, class_ids, target, threshold)


def prediction_to_logits(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = F.one_hot(pred.clamp_min(0), num_classes=num_classes).permute(0, 1, 4, 2, 3).float()
    return one_hot * 20.0


def evaluate_subset(
    model: VGAlignSegV4PartTransfer,
    loader: DataLoader,
    cache_dir: Path,
    device: str,
    num_classes: int,
    thresholds: list[float],
    source_topk: int = 1,
    logit_merge: str = "max",
    copy_source_views: bool = False,
    max_objects: int | None = None,
    output_dir: Path | None = None,
    viz_samples: int = 0,
    log_every: int = 25,
) -> Dict[str, float]:
    model.eval()
    totals_by_threshold = {
        threshold: {"pixel_errors": 0, "pixel_count": 0, "mean_iou_sum": 0.0, "batches": 0}
        for threshold in thresholds
    }
    viz_records = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_objects is not None and batch_idx >= max_objects:
                break
            if log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % log_every == 0):
                print(f"eval progress: object {batch_idx + 1}/{len(loader)}", flush=True)
            images = batch["images"].to(device, non_blocking=True)
            target = batch["masks"].to(device=device, dtype=torch.long, non_blocking=True)
            cached = cached_backbone_outputs(model, images, batch["object_id"], cache_dir, device)
            actor_logits, class_ids = actor_logits_for_object(
                model=model,
                images=images,
                target=target,
                actor_ids=batch["actor_ids"][0],
                cached=cached,
                num_classes=num_classes,
                source_topk=source_topk,
                logit_merge=logit_merge,
                copy_source_views=copy_source_views,
            )
            viz_preds_by_threshold = {} if output_dir is not None and batch_idx < viz_samples else None
            for threshold in thresholds:
                pred = compose_from_actor_logits(actor_logits, class_ids, target, threshold)
                metrics = segmentation_metrics_from_prediction(pred, target, num_classes)
                totals = totals_by_threshold[threshold]
                totals["pixel_errors"] += int(metrics["pixel_errors"])
                totals["pixel_count"] += int(target.numel())
                totals["mean_iou_sum"] += float(metrics["mean_iou"])
                totals["batches"] += 1
                if viz_preds_by_threshold is not None:
                    viz_preds_by_threshold[threshold] = pred.detach().cpu().to(torch.uint8)

            if viz_preds_by_threshold is not None:
                viz_records.append(
                    {
                        "batch_idx": batch_idx,
                        "object_id": batch["object_id"][0],
                        "view_names": batch["view_names"][0],
                        "images": images.detach().cpu(),
                        "target": target.detach().cpu(),
                        "preds_by_threshold": viz_preds_by_threshold,
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
    if best_metrics is None:
        return {"eval_pixel_accuracy": 0.0, "eval_mean_iou": 0.0, "eval_pixel_errors": 0, "eval_batches": 0}
    if output_dir is not None and viz_records:
        output_dir.mkdir(parents=True, exist_ok=True)
        for record in viz_records:
            viz_pred = record["preds_by_threshold"][best_threshold].to(dtype=torch.long)
            logits = prediction_to_logits(viz_pred, num_classes)
            save_visualization(
                output_dir / f"{record['batch_idx']:04d}_{record['object_id']}.png",
                record["images"],
                record["target"],
                logits,
                record["view_names"],
            )
        (output_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "best_threshold": best_threshold,
                    "eval_mean_iou": best_metrics["eval_mean_iou"],
                    "eval_pixel_accuracy": best_metrics["eval_pixel_accuracy"],
                    "threshold_metrics": threshold_metrics,
                },
                indent=2,
            )
        )
    return {**best_metrics, "eval_best_threshold": best_threshold, "eval_threshold_metrics": threshold_metrics}


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    seed_everything(args.seed + rank)
    random.seed(args.seed + rank)
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for cached object-level V4 training.")

    if is_rank0(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)

    train_set = ResultMultiViewDataset(
        args.data_root,
        split="train",
        views=args.views,
        image_size=args.image_size,
        train_count=optional_count(args.train_count),
        val_count=optional_count(args.val_count),
        test_count=optional_count(args.test_count),
        max_samples=args.max_train_samples,
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    eval_set = ResultMultiViewDataset(
        args.data_root,
        split=args.eval_split,
        views=args.views,
        image_size=args.image_size,
        train_count=optional_count(args.train_count),
        val_count=optional_count(args.val_count),
        test_count=optional_count(args.test_count),
        max_samples=args.max_val_samples,
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    train_sampler = (
        DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
        if distributed
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(args.num_workers, 2)),
        pin_memory=True,
        collate_fn=collate_batch,
    )

    meta = {
        "model_version": "v4_part_transfer",
        "num_classes": train_set.num_classes,
        "classes": train_set.classes,
        "train_samples": len(train_set),
        "eval_split": args.eval_split,
        "eval_samples": len(eval_set),
        "views": args.views,
        "image_size": args.image_size,
        "shuffle_split": args.shuffle_split,
        "split_seed": args.split_seed,
        "distributed": distributed,
        "world_size": world_size,
    }
    if is_rank0(rank):
        (args.output_dir / "run_config.json").write_text(json.dumps({"args": vars(args), "meta": meta}, indent=2, default=str))
        train_set.write_manifest(args.output_dir / "train_manifest.json")
        eval_set.write_manifest(args.output_dir / f"{args.eval_split}_manifest.json")
    barrier(distributed)

    base_model = VGAlignSegV4PartTransfer(
        checkpoint_path=args.checkpoint_path,
        hidden_dim=args.hidden_dim,
        refinement_iters=args.refinement_iters,
    ).to(device)
    resume_checkpoint: dict = {}
    if args.resume is not None:
        resume_checkpoint = load_trainable_state(base_model, args.resume, device)
        print_rank0(rank, f"Resumed V4 from {args.resume}", flush=True)

    base_model.backbone.eval()

    if args.eval_only:
        eval_thresholds = parse_thresholds(args)
        if is_rank0(rank):
            eval_dir = args.output_dir / f"{args.eval_split}_viz_eval_only"
            eval_metrics = evaluate_subset(
                base_model,
                eval_loader,
                args.cache_dir,
                device,
                train_set.num_classes,
                thresholds=eval_thresholds,
                source_topk=args.eval_source_topk,
                logit_merge=args.eval_logit_merge,
                copy_source_views=args.eval_copy_source_views,
                max_objects=args.eval_max_objects,
                output_dir=eval_dir,
                viz_samples=args.eval_viz_samples,
                log_every=args.eval_log_every,
            )
            metrics_path = args.output_dir / "metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": 0, "phase": args.eval_split, "epoch": 0, **eval_metrics}) + "\n")
            print(
                f"{args.eval_split}@eval_only: miou={eval_metrics['eval_mean_iou']:.4f} "
                f"thr={eval_metrics.get('eval_best_threshold', args.compose_threshold):.3f} "
                f"acc={eval_metrics['eval_pixel_accuracy']:.4f}",
                flush=True,
            )
        barrier(distributed)
        cleanup_distributed(distributed)
        return

    params = [p for name, p in base_model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_optimizer and resume_checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)

    train_model = (
        DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if device.startswith("cuda") else None,
            output_device=local_rank if device.startswith("cuda") else None,
            find_unused_parameters=False,
        )
        if distributed
        else base_model
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    eval_thresholds = parse_thresholds(args)
    train_fixed_source_views = parse_view_ids(args.train_fixed_source_views)
    train_loss_views = parse_view_ids(args.train_loss_views)
    global_step = int(resume_checkpoint.get("step", 0)) if args.resume_optimizer else 0
    best_eval_miou = -1.0
    print_rank0(
        rank,
        f"Training VG-AlignSeg V4 part transfer: train={len(train_set)} {args.eval_split}={len(eval_set)} "
        f"classes={train_set.num_classes} cache={args.cache_dir} world_size={world_size}",
        flush=True,
    )

    steps_per_epoch = max(1, len(train_loader))
    epoch = global_step // steps_per_epoch + 1
    while args.epochs <= 0 or epoch <= args.epochs:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_model.train()
        base_model.backbone.eval()
        for batch in train_loader:
            global_step += 1
            images = batch["images"].to(device, non_blocking=True)
            prompt = choose_training_prompt(
                batch,
                device,
                source_policy=args.train_source_policy,
                source_topk=args.train_source_topk,
                fixed_source_views=train_fixed_source_views,
            )
            target = prompt["target"].to(device=device, dtype=torch.float32)
            sources = [
                (
                    source_mask.to(device=device, dtype=torch.float32),
                    source_view.to(device=device, dtype=torch.long),
                )
                for source_mask, source_view in prompt["sources"]
            ]
            cached = cached_backbone_outputs(base_model, images, batch["object_id"], args.cache_dir, device)

            optimizer.zero_grad(set_to_none=True)
            source_view_ids = [int(source_view.item()) for _, source_view in sources]

            def compute_losses(logits: torch.Tensor, boundary_logits: torch.Tensor, active_source_views: list[int]):
                loss_views = select_loss_views(
                    target.shape[1],
                    train_loss_views,
                    active_source_views,
                    args.train_exclude_source_views_from_loss,
                )
                if loss_views:
                    logits_for_loss = logits[:, loss_views]
                    boundary_logits_for_loss = boundary_logits[:, loss_views]
                    target_for_loss = target[:, loss_views]
                else:
                    logits_for_loss = logits
                    boundary_logits_for_loss = boundary_logits
                    target_for_loss = target
                pos = target_for_loss.sum().clamp_min(1.0)
                neg = (target_for_loss.numel() - target_for_loss.sum()).clamp_min(1.0)
                pos_weight = (neg / pos).clamp(1.0, 30.0)
                bce_i = F.binary_cross_entropy_with_logits(logits_for_loss, target_for_loss, pos_weight=pos_weight)
                focal_i = binary_focal_loss(logits_for_loss, target_for_loss, alpha=args.focal_alpha, gamma=args.focal_gamma)
                dice_i = binary_dice_loss(logits_for_loss, target_for_loss)
                tversky_i = binary_tversky_loss(logits_for_loss, target_for_loss)
                boundary_i = binary_boundary_loss(logits_for_loss, target_for_loss)
                boundary_aux_i = F.binary_cross_entropy_with_logits(
                    boundary_logits_for_loss,
                    binary_boundary_target(target_for_loss),
                )
                loss_i = (
                    args.bce_loss_weight * bce_i
                    + args.focal_loss_weight * focal_i
                    + args.dice_loss_weight * dice_i
                    + args.tversky_loss_weight * tversky_i
                    + args.boundary_loss_weight * boundary_i
                    + args.boundary_head_loss_weight * boundary_aux_i
                )
                return loss_i, bce_i, focal_i, dice_i, tversky_i, boundary_i, boundary_aux_i, logits_for_loss, target_for_loss

            if args.train_independent_sources:
                loss_items = []
                metric_logits = None
                metric_target = None
                full_logits_for_viz = None
                for source_mask, source_view in sources:
                    outputs = train_model(images, source_mask, source_view, cached, images.shape[-2:])
                    one_logits = outputs["binary_logits"][:, :, 0]
                    one_boundary_logits = outputs["boundary_logits"][:, :, 0]
                    full_logits_for_viz = one_logits.detach()
                    losses = compute_losses(one_logits, one_boundary_logits, [int(source_view.item())])
                    (loss_i, bce_i, focal_i, dice_i, tversky_i, boundary_i, boundary_aux_i, logits_i, target_i) = losses
                    (loss_i / max(1, len(sources))).backward()
                    loss_items.append(
                        (
                            loss_i.detach(),
                            bce_i.detach(),
                            focal_i.detach(),
                            dice_i.detach(),
                            tversky_i.detach(),
                            boundary_i.detach(),
                            boundary_aux_i.detach(),
                        )
                    )
                    metric_logits = logits_i.detach()
                    metric_target = target_i
                loss, bce, focal, dice, tversky, boundary, boundary_aux = [
                    torch.stack([items[idx] for items in loss_items]).mean()
                    for idx in range(7)
                ]
                logits_for_loss = metric_logits if metric_logits is not None else torch.zeros_like(target[:, :1])
                target_for_loss = metric_target if metric_target is not None else target[:, :1]
                logits = full_logits_for_viz if full_logits_for_viz is not None else torch.zeros_like(target)
            else:
                outputs_by_source = [
                    train_model(images, source_mask, source_view, cached, images.shape[-2:])
                    for source_mask, source_view in sources
                ]
                logits, boundary_logits = merge_source_outputs(
                    outputs_by_source,
                    sources,
                    logit_merge=args.train_logit_merge,
                    copy_source_views=args.train_copy_source_views,
                )
                losses = compute_losses(logits, boundary_logits, source_view_ids)
                loss, bce, focal, dice, tversky, boundary, boundary_aux, logits_for_loss, target_for_loss = losses
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            if global_step == 1 or global_step % args.log_every == 0:
                with torch.no_grad():
                    bm = binary_metrics(logits_for_loss.detach(), target_for_loss)
                pixel_errors = reduce_sum(bm["binary_errors"], device, distributed)
                pixel_count = reduce_sum(target_for_loss.numel(), device, distributed)
                row = {
                    "step": global_step,
                    "phase": "train",
                    "epoch": epoch,
                    "object_id": batch["object_id"][0],
                    "actor_id": int(prompt["actor_id"]),
                    "source_view": int(prompt["source_views"][0]),
                    "source_views": prompt["source_views"],
                    "loss": reduce_mean(loss.item(), device, distributed),
                    "bce": reduce_mean(bce.item(), device, distributed),
                    "focal": reduce_mean(focal.item(), device, distributed),
                    "dice": reduce_mean(dice.item(), device, distributed),
                    "tversky": reduce_mean(tversky.item(), device, distributed),
                    "boundary": reduce_mean(boundary.item(), device, distributed),
                    "boundary_aux": reduce_mean(boundary_aux.item(), device, distributed),
                    "binary_iou": reduce_mean(bm["binary_iou"], device, distributed),
                    "binary_accuracy": 1.0 - pixel_errors / max(1.0, pixel_count),
                    "binary_errors": int(pixel_errors),
                }
                if is_rank0(rank):
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        f"step={global_step:06d} epoch={epoch:03d} loss={row['loss']:.4f} "
                        f"actor={row['actor_id']} src={row['source_views']} "
                        f"biou={row['binary_iou']:.4f} bacc={row['binary_accuracy']:.4f}",
                        flush=True,
                    )

            if is_rank0(rank) and (global_step == 1 or global_step % args.viz_every == 0):
                pseudo_target = target.long()
                pseudo_pred_logits = torch.stack([-logits.detach(), logits.detach()], dim=2)
                save_visualization(
                    args.output_dir / f"train_transfer_step_{global_step:06d}.png",
                    images.detach().cpu(),
                    pseudo_target.detach().cpu(),
                    pseudo_pred_logits.detach().cpu(),
                    batch["view_names"][0],
                )

            if global_step % args.save_every == 0:
                barrier(distributed)
                if is_rank0(rank):
                    eval_dir = args.output_dir / f"{args.eval_split}_viz_step_{global_step:06d}"
                    eval_metrics = evaluate_subset(
                        base_model,
                        eval_loader,
                        args.cache_dir,
                        device,
                        train_set.num_classes,
                        thresholds=eval_thresholds,
                        source_topk=args.eval_source_topk,
                        logit_merge=args.eval_logit_merge,
                        copy_source_views=args.eval_copy_source_views,
                        max_objects=args.eval_max_objects,
                        output_dir=eval_dir,
                        viz_samples=args.eval_viz_samples,
                        log_every=args.eval_log_every,
                    )
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": global_step, "phase": args.eval_split, "epoch": epoch, **eval_metrics}) + "\n")
                    save_checkpoint(args.output_dir / "latest.pt", base_model, optimizer, global_step, args, meta)
                    score = eval_metrics["eval_mean_iou"]
                    if score > best_eval_miou:
                        best_eval_miou = score
                        save_checkpoint(args.output_dir / "best.pt", base_model, optimizer, global_step, args, meta)
                    print(
                        f"{args.eval_split}@{global_step}: miou={eval_metrics['eval_mean_iou']:.4f} "
                        f"thr={eval_metrics.get('eval_best_threshold', args.compose_threshold):.3f} "
                        f"acc={eval_metrics['eval_pixel_accuracy']:.4f}",
                        flush=True,
                    )
                    stop_for_target = bool(args.target_eval_miou is not None and score >= args.target_eval_miou)
                    if stop_for_target:
                        print(f"Reached target_eval_miou={args.target_eval_miou:.4f} with score={score:.4f}; stopping.", flush=True)
                else:
                    stop_for_target = False
                if distributed:
                    flag = torch.tensor(1 if stop_for_target else 0, device=device)
                    dist.broadcast(flag, src=0)
                    stop_for_target = bool(flag.item())
                barrier(distributed)
                if stop_for_target:
                    cleanup_distributed(distributed)
                    return
                train_model.train()
                base_model.backbone.eval()

            if args.max_steps is not None and global_step >= args.max_steps:
                if is_rank0(rank):
                    save_checkpoint(args.output_dir / "latest.pt", base_model, optimizer, global_step, args, meta)
                    print(f"Reached max_steps={args.max_steps}; saved latest checkpoint.", flush=True)
                cleanup_distributed(distributed)
                return
        epoch += 1

    if is_rank0(rank):
        save_checkpoint(args.output_dir / "latest.pt", base_model, optimizer, global_step, args, meta)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
