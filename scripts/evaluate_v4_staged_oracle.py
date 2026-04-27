from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset  # noqa: E402
from models import VGAlignSegV4PartTransfer  # noqa: E402
from scripts.eval_utils import save_visualization, segmentation_metrics_from_prediction  # noqa: E402
from scripts.train_v2_result import cached_backbone_outputs, collate_batch, seed_everything  # noqa: E402
from scripts.train_v4_part_transfer import actor_logits_for_object, compose_from_actor_logits, prediction_to_logits  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the staged/oracle V4 protocol with source GT copy and category summaries."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--category-list", type=Path, default=REPO_ROOT / "data" / "hf_metadata" / "category_models_list.txt")
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
    parser.add_argument("--copy-source-views", action="store_true")
    parser.add_argument("--viz-samples", type=int, default=80)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def optional_count(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def load_category_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    mapping: dict[str, str] = {}
    category = None
    category_pattern = re.compile(r"^.*?([A-Za-z][A-Za-z0-9]*)\D+(\d+).*?$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") or line.startswith("【"):
            match = category_pattern.match(line)
            if match:
                category = match.group(1)
            continue
        if line.startswith("-") and category is not None:
            object_id = line.split("-", 1)[1].strip()
            if object_id:
                mapping[object_id] = category
    return mapping


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


def actor_count_bucket(actor_count: int) -> str:
    if actor_count <= 2:
        return "coarse_1_2_parts"
    if actor_count <= 4:
        return "medium_3_4_parts"
    return "fine_5plus_parts"


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_groups(per_object: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in per_object:
        groups[str(row[key])].append(float(row["mean_iou"]))
    return {
        group: {
            "objects": len(values),
            "mean_iou": mean(values),
        }
        for group, values in sorted(groups.items())
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    thresholds = parse_floats(args.thresholds)
    category_map = load_category_map(args.category_list)
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
        threshold: {"pixel_errors": 0, "pixel_count": 0, "mean_iou_sum": 0.0, "objects": 0}
        for threshold in thresholds
    }
    per_object_by_threshold: dict[float, list[dict]] = {threshold: [] for threshold in thresholds}
    viz_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % args.log_every == 0):
                print(f"eval progress: object {batch_idx + 1}/{len(loader)}", flush=True)
            images = batch["images"].to(args.device, non_blocking=True)
            target = batch["masks"].to(device=args.device, dtype=torch.long, non_blocking=True)
            object_id = str(batch["object_id"][0])
            actor_ids = [int(actor_id) for actor_id in batch["actor_ids"][0]]
            cached = cached_backbone_outputs(model, images, batch["object_id"], args.cache_dir, args.device)
            actor_logits, class_ids = actor_logits_for_object(
                model=model,
                images=images,
                target=target,
                actor_ids=actor_ids,
                cached=cached,
                num_classes=num_classes,
                source_topk=args.source_topk,
                logit_merge=args.logit_merge,
                copy_source_views=args.copy_source_views,
            )

            preds_by_threshold = {}
            for threshold in thresholds:
                pred = compose_from_actor_logits(actor_logits, class_ids, target, threshold)
                metrics = segmentation_metrics_from_prediction(pred, target, num_classes)
                total = totals[threshold]
                total["pixel_errors"] += int(metrics["pixel_errors"])
                total["pixel_count"] += int(target.numel())
                total["mean_iou_sum"] += float(metrics["mean_iou"])
                total["objects"] += 1
                per_object_by_threshold[threshold].append(
                    {
                        "index": batch_idx,
                        "object_id": object_id,
                        "category": category_map.get(object_id, "Unknown"),
                        "actor_count": len(actor_ids),
                        "granularity": actor_count_bucket(len(actor_ids)),
                        "mean_iou": float(metrics["mean_iou"]),
                        "pixel_accuracy": float(metrics["pixel_accuracy"]),
                        "pixel_errors": int(metrics["pixel_errors"]),
                    }
                )
                if batch_idx < args.viz_samples:
                    preds_by_threshold[threshold] = pred.detach().cpu().to(torch.uint8)

            if batch_idx < args.viz_samples:
                viz_records.append(
                    {
                        "batch_idx": batch_idx,
                        "object_id": object_id,
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
            "eval_batches": total["objects"],
        }
        threshold_metrics[str(threshold)] = metrics
        if best_metrics is None or metrics["eval_mean_iou"] > best_metrics["eval_mean_iou"]:
            best_threshold = threshold
            best_metrics = metrics

    assert best_metrics is not None
    per_object = per_object_by_threshold[best_threshold]
    per_category = summarize_groups(per_object, "category")
    per_granularity = summarize_groups(per_object, "granularity")
    category_values = [row["mean_iou"] for row in per_category.values()]
    granularity_values = [row["mean_iou"] for row in per_granularity.values()]
    table_metrics = {
        "iou_object_category": mean(category_values),
        "iou_granularity": mean(granularity_values),
        "iou_part": float(best_metrics["eval_mean_iou"]),
        "cross_view_consistency_acc": float(best_metrics["eval_pixel_accuracy"]),
    }

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
        "copy_source_views": args.copy_source_views,
        "metric": "full 8-view staged/oracle segmentation with source masks copied when enabled",
        "best_threshold": best_threshold,
        **best_metrics,
        "table_metrics": table_metrics,
        "threshold_metrics": threshold_metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "per_object.json").write_text(json.dumps(per_object, indent=2), encoding="utf-8")
    (args.output_dir / "per_category.json").write_text(json.dumps(per_category, indent=2), encoding="utf-8")
    (args.output_dir / "per_granularity.json").write_text(json.dumps(per_granularity, indent=2), encoding="utf-8")
    dataset.write_manifest(args.output_dir / f"{args.split}_manifest.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

