from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset, square_pad_image  # noqa: E402
from scripts.eval_utils import save_visualization  # noqa: E402
from scripts.train_v2_result import collate_batch, seed_everything  # noqa: E402
from scripts.train_v4_part_transfer import prediction_to_logits  # noqa: E402


DEFAULT_CLASS_PROMPTS = [
    "object",
    "part",
    "background",
    "chair",
    "seat",
    "back",
    "leg",
    "base",
    "arm",
    "bottle",
    "cap",
    "body",
    "box",
    "clock",
    "display",
    "screen",
    "door",
    "handle",
    "faucet",
    "tap",
    "keyboard",
    "key",
    "laptop",
    "microwave",
    "oven",
    "storage furniture",
    "cabinet",
    "shelf",
    "drawer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PanSt3R on the VG-AlignSeg test split with class-agnostic "
            "oracle matching from panoptic segments to GT actors."
        )
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/home/lyx/curriculum/computer_vision/baseline_weights/panst3r/panst3r_v2_512_5ds.pth"),
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "final_results" / "baselines" / "panst3r" / "v2_224_oracle_match",
    )
    parser.add_argument("--category-map", type=Path, default=REPO_ROOT / "final_results" / "metadata" / "object_category_map.json")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=-1)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--viz-samples", type=int, default=40)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--max-bs", type=int, default=1)
    parser.add_argument("--num-keyframes", type=int, default=8)
    parser.add_argument("--postprocess", choices=("standard_v2", "qubo"), default="standard_v2")
    parser.add_argument(
        "--dynamic-vocab",
        action="store_true",
        help="Re-encode text prompts for every object. Default pre-encodes prompts once as a fixed vocabulary.",
    )
    parser.add_argument(
        "--class-prompts",
        type=str,
        default="",
        help="Comma-separated prompts. Defaults to VG categories plus generic part names.",
    )
    parser.add_argument("--seed", type=int, default=19)
    return parser.parse_args()


def optional_count(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def actor_count_bucket(actor_count: int) -> str:
    if actor_count <= 2:
        return "coarse_1_2_parts"
    if actor_count <= 4:
        return "medium_3_4_parts"
    return "fine_5plus_parts"


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def summarize_groups(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(float(row["mean_iou"]))
    return {
        name: {
            "objects": len(values),
            "mean_iou": mean(values),
        }
        for name, values in sorted(groups.items())
    }


def load_category_map(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_class_prompts(category_map: dict[str, str], extra: str) -> list[str]:
    prompts = list(DEFAULT_CLASS_PROMPTS)
    prompts.extend(category_map.values())
    if extra.strip():
        prompts.extend(item.strip() for item in extra.split(",") if item.strip())
    deduped = []
    seen = set()
    for prompt in prompts:
        normalized = str(prompt).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def load_panst3r_image(path: Path, image_size: int) -> torch.Tensor:
    from must3r.datasets import ImgNorm

    img = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(background, img).convert("RGB")
    img = square_pad_image(img, (255, 255, 255))
    img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    return ImgNorm(img).contiguous()


def run_panst3r(
    model,
    image_paths: Sequence[Path],
    classes: Sequence[str],
    image_size: int,
    device: str,
    amp: str,
    max_bs: int,
    num_keyframes: int,
    postprocess: str,
) -> torch.Tensor:
    from panst3r.engine import panoptic_inference_qubo, panoptic_inference_v2

    imgs = [load_panst3r_image(path, image_size).to(device) for path in image_paths]
    true_shape = torch.tensor([[image_size, image_size] for _ in imgs], dtype=torch.int32, device=device)
    amp_value: bool | str = False if amp == "none" else amp
    out_3d, pan_out = model.forward_inference_multi_ar(
        imgs,
        true_shape,
        list(classes),
        num_keyframes=min(num_keyframes, len(imgs)),
        use_retrieval=False,
        max_bs=max_bs,
        outdevice=device,
        amp=amp_value,
    )
    del out_3d
    size = true_shape.cpu().numpy()
    label_mode = model.panoptic_decoder.label_mode
    if postprocess == "qubo":
        pan_preds = panoptic_inference_qubo(
            pan_out["pred_logits"],
            pan_out["pred_masks"],
            size,
            label_mode=label_mode,
            device="cpu",
            multi_ar=True,
            silent=True,
        )
    else:
        pan_preds = panoptic_inference_v2(
            pan_out["pred_logits"],
            pan_out["pred_masks"],
            size,
            label_mode=label_mode,
            device="cpu",
            multi_ar=True,
        )
    pred = torch.stack([view.to(torch.long) for view in pan_preds[0]["pan"]], dim=0)
    return pred


def oracle_match_prediction(pred_segments: torch.Tensor, target: torch.Tensor, class_ids: Sequence[int]) -> tuple[torch.Tensor, dict]:
    pred_segments = pred_segments.to(torch.long).cpu()
    target = target.to(torch.long).cpu()
    pred_ids = [int(seg_id) for seg_id in torch.unique(pred_segments).tolist() if int(seg_id) != 0]
    class_ids = [int(class_id) for class_id in class_ids]
    if not class_ids:
        return torch.zeros_like(target), {"mean_iou": 0.0, "per_class": {}, "matched_pairs": []}

    if pred_ids:
        iou_matrix = np.zeros((len(class_ids), len(pred_ids)), dtype=np.float64)
        for row, class_id in enumerate(class_ids):
            gt_mask = target == class_id
            for col, pred_id in enumerate(pred_ids):
                pred_mask = pred_segments == pred_id
                union = (gt_mask | pred_mask).sum().item()
                if union > 0:
                    inter = (gt_mask & pred_mask).sum().item()
                    iou_matrix[row, col] = float(inter) / float(union)
        rows, cols = linear_sum_assignment(-iou_matrix)
    else:
        iou_matrix = np.zeros((len(class_ids), 0), dtype=np.float64)
        rows = np.array([], dtype=np.int64)
        cols = np.array([], dtype=np.int64)

    matched_by_class = {}
    matched_pairs = []
    mapped = torch.zeros_like(target)
    for row, col in zip(rows.tolist(), cols.tolist()):
        iou = float(iou_matrix[row, col])
        class_id = class_ids[row]
        pred_id = pred_ids[col]
        matched_by_class[class_id] = {"pred_segment": pred_id, "iou": iou}
        matched_pairs.append({"class_id": class_id, "pred_segment": pred_id, "iou": iou})
        mapped[pred_segments == pred_id] = class_id

    per_class = {}
    ious = []
    for class_id in class_ids:
        gt_mask = target == class_id
        pred_mask = mapped == class_id
        union = (gt_mask | pred_mask).sum().item()
        inter = (gt_mask & pred_mask).sum().item()
        iou = float(inter) / float(union) if union > 0 else 0.0
        ious.append(iou)
        match = matched_by_class.get(class_id)
        per_class[str(class_id)] = {
            "iou": iou,
            "gt_pixels": int(gt_mask.sum().item()),
            "pred_pixels": int(pred_mask.sum().item()),
            "matched_pred_segment": None if match is None else int(match["pred_segment"]),
        }
    pixel_errors = int((mapped != target).sum().item())
    pixel_count = int(target.numel())
    return mapped, {
        "mean_iou": mean(ious),
        "pixel_accuracy": 1.0 - pixel_errors / max(1, pixel_count),
        "pixel_errors": pixel_errors,
        "pixel_count": pixel_count,
        "actors": len(class_ids),
        "pred_segments": len(pred_ids),
        "per_class": per_class,
        "matched_pairs": matched_pairs,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    from panst3r import PanSt3R

    category_map = load_category_map(args.category_map)
    classes = build_class_prompts(category_map, args.class_prompts)
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
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_batch)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading PanSt3R from {args.weights}", flush=True)
    model = PanSt3R.from_checkpoint(args.weights)
    if hasattr(model.panoptic_decoder, "text_encoder"):
        if args.dynamic_vocab:
            model.panoptic_decoder.text_encoder.change_mode(fixed_vocab=False)
        else:
            print("Encoding PanSt3R class prompts once as a fixed vocabulary.", flush=True)
            model.panoptic_decoder.text_encoder.set_vocab(classes, device=args.device)
            model.panoptic_decoder.text_encoder.change_mode(fixed_vocab=True)
    model = model.to(args.device).eval()
    print(f"Loaded PanSt3R; evaluating {len(dataset)} objects with {len(classes)} prompts.", flush=True)

    per_object = []
    viz_records = []
    pixel_errors = 0
    pixel_count = 0
    mean_iou_sum = 0.0
    actor_instances = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if args.max_objects is not None and batch_idx >= args.max_objects:
                break
            object_id = str(batch["object_id"][0])
            if args.log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % args.log_every == 0):
                print(f"eval progress: object {batch_idx + 1}/{len(dataset)} ({object_id})", flush=True)

            obj_root = Path(batch["root"][0]) if "root" in batch else Path(args.data_root) / f"{object_id}_views"
            image_paths = [obj_root / "color" / name for name in batch["view_names"][0]]
            pred_segments = run_panst3r(
                model=model,
                image_paths=image_paths,
                classes=classes,
                image_size=args.image_size,
                device=args.device,
                amp=args.amp,
                max_bs=args.max_bs,
                num_keyframes=args.num_keyframes,
                postprocess=args.postprocess,
            )
            target = batch["masks"][0].to(torch.long)
            class_ids = [int(actor_id) - 1 for actor_id in batch["actor_ids"][0]]
            mapped_pred, metrics = oracle_match_prediction(pred_segments, target, class_ids)

            mean_iou_sum += float(metrics["mean_iou"])
            pixel_errors += int(metrics["pixel_errors"])
            pixel_count += int(metrics["pixel_count"])
            actor_instances += int(metrics["actors"])
            row = {
                "index": batch_idx,
                "object_id": object_id,
                "category": category_map.get(object_id, "Unknown"),
                "actor_count": len(class_ids),
                "granularity": actor_count_bucket(len(class_ids)),
                "mean_iou": float(metrics["mean_iou"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
                "pixel_errors": int(metrics["pixel_errors"]),
                "actors": int(metrics["actors"]),
                "pred_segments": int(metrics["pred_segments"]),
                "matched_pairs": metrics["matched_pairs"],
                "per_class": metrics["per_class"],
            }
            per_object.append(row)

            if batch_idx < args.viz_samples:
                viz_records.append(
                    {
                        "batch_idx": batch_idx,
                        "object_id": object_id,
                        "view_names": batch["view_names"][0],
                        "images": batch["images"],
                        "target": batch["masks"],
                        "pred": mapped_pred.unsqueeze(0),
                    }
                )

            del pred_segments, mapped_pred
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    eval_batches = len(per_object)
    per_category = summarize_groups(per_object, "category")
    per_granularity = summarize_groups(per_object, "granularity")
    table_metrics = {
        "iou_object_category": mean([row["mean_iou"] for row in per_category.values()]),
        "iou_granularity": mean([row["mean_iou"] for row in per_granularity.values()]),
        "iou_part": mean_iou_sum / max(1, eval_batches),
        "cross_view_consistency_acc": 1.0 - pixel_errors / max(1, pixel_count),
    }
    summary = {
        "baseline": "PanSt3R",
        "repository": "https://github.com/naver/panst3r",
        "checkpoint": str(args.weights),
        "protocol": "class-agnostic oracle matching from PanSt3R panoptic segments to VG-AlignSeg GT actor IDs",
        "metric_note": (
            "PanSt3R predicts category-level panoptic segments, not ShapeNetPart actor IDs. "
            "Hungarian matching is used only for evaluation label assignment; no GT mask is copied into predictions."
        ),
        "split": args.split,
        "shuffle_split": args.shuffle_split,
        "split_seed": args.split_seed,
        "train_count": args.train_count,
        "test_count": eval_batches,
        "image_size": args.image_size,
        "views": args.views,
        "postprocess": args.postprocess,
        "amp": args.amp,
        "num_keyframes": args.num_keyframes,
        "dynamic_vocab": args.dynamic_vocab,
        "class_prompts": classes,
        "eval_mean_iou": table_metrics["iou_part"],
        "eval_pixel_accuracy": table_metrics["cross_view_consistency_acc"],
        "eval_pixel_errors": pixel_errors,
        "eval_pixel_count": pixel_count,
        "eval_actor_instances": actor_instances,
        "eval_batches": eval_batches,
        "table_metrics": table_metrics,
    }

    viz_dir = args.output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    num_classes = dataset.num_classes
    for record in viz_records:
        logits = prediction_to_logits(record["pred"].to(torch.long), num_classes)
        save_visualization(
            viz_dir / f"{record['batch_idx']:04d}_{record['object_id']}.png",
            record["images"],
            record["target"],
            logits,
            record["view_names"],
        )

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "per_object.json").write_text(json.dumps(per_object, indent=2), encoding="utf-8")
    (args.output_dir / "per_category.json").write_text(json.dumps(per_category, indent=2), encoding="utf-8")
    (args.output_dir / "per_granularity.json").write_text(json.dumps(per_granularity, indent=2), encoding="utf-8")
    dataset.write_manifest(args.output_dir / f"{args.split}_manifest.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
