from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import MiniBatchKMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset, VIEW_ORDER, square_pad_image  # noqa: E402
from data.views_36845 import colorize  # noqa: E402
from scripts.evaluate_panst3r_baseline import (  # noqa: E402
    actor_count_bucket,
    load_category_map,
    mean,
    oracle_match_prediction,
    summarize_groups,
)
from scripts.train_v2_result import seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate transparent 2D VG-AlignSeg adapters for PartSLIP2 and COPS, "
            "and generate paper-ready comparison figures."
        )
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "final_results" / "baselines" / "partslip2_cops_visual_adapters",
    )
    parser.add_argument("--category-map", type=Path, default=REPO_ROOT / "final_results" / "metadata" / "object_category_map.json")
    parser.add_argument("--partslip2-repo", type=Path, default=REPO_ROOT.parent / "PartSLIP2")
    parser.add_argument("--cops-repo", type=Path, default=REPO_ROOT.parent / "COPS")
    parser.add_argument("--v4-viz-dir", type=Path, default=REPO_ROOT / "final_results" / "evaluations" / "v4_target_only_top4_best4500" / "visualizations")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=-1)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--viz-samples", type=int, default=10)
    parser.add_argument("--viz-random", action="store_true", help="Randomly choose visualization objects from the evaluated split.")
    parser.add_argument("--viz-seed", type=int, default=20260429)
    parser.add_argument(
        "--viz-indices",
        type=str,
        default="",
        help="Comma-separated evaluated object indices to visualize. Overrides --viz-random and --viz-samples.",
    )
    parser.add_argument("--paper-views", type=str, default="0,2,4,6")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--max-cluster-pixels", type=int, default=12000)
    return parser.parse_args()


def optional_count(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=path, text=True).strip()
    except Exception:
        return None


def load_rgba_rgb_alpha(path: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    rgba = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    rgb = Image.alpha_composite(bg, rgba).convert("RGB")
    alpha = rgba.getchannel("A")
    rgb = square_pad_image(rgb, (255, 255, 255))
    alpha = square_pad_image(alpha, (0,))
    rgb = rgb.resize((image_size, image_size), Image.Resampling.BICUBIC)
    alpha = alpha.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(rgb, dtype=np.uint8), np.asarray(alpha, dtype=np.uint8) > 8


def load_object_rgb_alpha(obj_root: Path, view_names: Sequence[str], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    rgbs = []
    alphas = []
    for name in view_names:
        rgb, alpha = load_rgba_rgb_alpha(obj_root / "color" / name, image_size)
        rgbs.append(rgb)
        alphas.append(alpha)
    return np.stack(rgbs, axis=0), np.stack(alphas, axis=0)


def deterministic_sample(num_points: int, max_points: int, seed: int) -> np.ndarray:
    if num_points <= max_points:
        return np.arange(num_points)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_points, size=max_points, replace=False))


def sort_clusters_by_centroid(labels: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    remap = np.zeros(k, dtype=np.int64)
    centroids = []
    for cid in range(k):
        pts = coords[labels == cid]
        if len(pts) == 0:
            centroids.append((1.0e6, 1.0e6, cid))
        else:
            yx = pts.mean(axis=0)
            centroids.append((float(yx[1]), float(yx[0]), cid))
    for new_id, (_, _, old_id) in enumerate(sorted(centroids), start=1):
        remap[old_id] = new_id
    return remap[labels]


def partslip2_2d_adapter(rgbs: np.ndarray, fg: np.ndarray, k: int, seed: int, max_pixels: int) -> np.ndarray:
    """Per-view RGB+XY proposals; a VG-AlignSeg 2D proxy, not native PartSLIP++."""
    views, height, width, _ = rgbs.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xy = np.stack([yy / max(1, height - 1), xx / max(1, width - 1)], axis=-1).astype(np.float32)
    pred = np.zeros((views, height, width), dtype=np.int64)
    for view_idx in range(views):
        mask = fg[view_idx]
        points = np.argwhere(mask)
        if len(points) == 0:
            continue
        local_k = min(k, len(points))
        rgb_feat = rgbs[view_idx][mask].astype(np.float32) / 255.0
        xy_feat = xy[mask] * 0.45
        features = np.concatenate([rgb_feat, xy_feat], axis=1)
        train_idx = deterministic_sample(len(features), max_pixels, seed + view_idx)
        model = MiniBatchKMeans(
            n_clusters=local_k,
            random_state=seed + view_idx,
            batch_size=2048,
            n_init=5,
            max_iter=80,
            reassignment_ratio=0.01,
        )
        model.fit(features[train_idx])
        labels = model.predict(features)
        sorted_labels = sort_clusters_by_centroid(labels, points.astype(np.float32), local_k)
        pred[view_idx][mask] = sorted_labels
    return pred


def cops_2d_adapter(rgbs: np.ndarray, fg: np.ndarray, k: int, seed: int, max_pixels: int) -> np.ndarray:
    """Global multi-view RGB+XY+view clustering; a VG-AlignSeg 2D proxy, not native COPS."""
    views, height, width, _ = rgbs.shape
    yy, xx = np.mgrid[0:height, 0:width]
    xy = np.stack([yy / max(1, height - 1), xx / max(1, width - 1)], axis=-1).astype(np.float32)
    features = []
    coords = []
    for view_idx in range(views):
        mask = fg[view_idx]
        if not mask.any():
            continue
        view_code = np.full((int(mask.sum()), 1), view_idx / max(1, views - 1), dtype=np.float32) * 0.12
        rgb_feat = rgbs[view_idx][mask].astype(np.float32) / 255.0
        xy_feat = xy[mask] * 0.35
        features.append(np.concatenate([rgb_feat, xy_feat, view_code], axis=1))
        view_col = np.full((int(mask.sum()), 1), view_idx, dtype=np.int64)
        coords.append(np.concatenate([view_col, np.argwhere(mask)], axis=1))
    pred = np.zeros((views, height, width), dtype=np.int64)
    if not features:
        return pred
    features_np = np.concatenate(features, axis=0)
    coords_np = np.concatenate(coords, axis=0)
    local_k = min(k, len(features_np))
    train_idx = deterministic_sample(len(features_np), max_pixels, seed)
    model = MiniBatchKMeans(
        n_clusters=local_k,
        random_state=seed,
        batch_size=4096,
        n_init=8,
        max_iter=120,
        reassignment_ratio=0.01,
    )
    model.fit(features_np[train_idx])
    labels = model.predict(features_np)
    sorted_labels = sort_clusters_by_centroid(labels, coords_np[:, 1:].astype(np.float32), local_k)
    pred[coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]] = sorted_labels
    return pred


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.48) -> Image.Image:
    base = rgb.astype(np.float32)
    color = np.asarray(colorize(mask.astype(np.int64)), dtype=np.float32)
    out = base.copy()
    fg = mask > 0
    out[fg] = (1.0 - alpha) * base[fg] + alpha * color[fg]
    return Image.fromarray(out.clip(0, 255).astype(np.uint8), mode="RGB")


def mask_image(mask: np.ndarray) -> Image.Image:
    return colorize(mask.astype(np.int64)).convert("RGB")


def save_mask_set(root: Path, object_tag: str, raw: np.ndarray, mapped: torch.Tensor, view_names: Sequence[str]) -> None:
    out = root / object_tag
    out.mkdir(parents=True, exist_ok=True)
    mapped_np = mapped.cpu().numpy().astype(np.int64)
    np.savez_compressed(out / "masks.npz", raw_segments=raw.astype(np.int16), mapped_prediction=mapped_np.astype(np.int16))
    for view_idx, name in enumerate(view_names):
        stem = Path(name).stem
        mask_image(raw[view_idx]).save(out / f"{stem}_raw.png")
        mask_image(mapped_np[view_idx]).save(out / f"{stem}_mapped.png")


def find_v4_viz(viz_dir: Path, index: int, object_id: str) -> Path | None:
    direct = viz_dir / f"{index:04d}_{object_id}.png"
    if direct.exists():
        return direct
    matches = sorted(viz_dir.glob(f"*_{object_id}.png"))
    return matches[0] if matches else None


def crop_v4_prediction(viz_path: Path | None, view_idx: int, image_size: int) -> Image.Image:
    if viz_path is None or not viz_path.exists():
        return Image.new("RGB", (image_size, image_size), (245, 245, 245))
    img = Image.open(viz_path).convert("RGB")
    label_h = 24
    row_h = image_size + label_h
    left = image_size * 2
    top = view_idx * row_h + label_h
    return img.crop((left, top, left + image_size, top + image_size))


def parse_paper_views(value: str, views: int) -> list[int]:
    out = []
    for part in value.split(","):
        if not part.strip():
            continue
        idx = int(part)
        if 0 <= idx < views:
            out.append(idx)
    return out or list(range(min(4, views)))


def parse_viz_indices(value: str, total: int) -> set[int] | None:
    if not value.strip():
        return None
    indices = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= total:
            raise ValueError(f"Visualization index {idx} is outside evaluated range [0, {total}).")
        indices.add(idx)
    return indices


def make_paper_figure(
    out_path: Path,
    object_id: str,
    rgbs: np.ndarray,
    target: torch.Tensor,
    v4_viz_path: Path | None,
    partslip_pred: torch.Tensor,
    cops_pred: torch.Tensor,
    view_names: Sequence[str],
    paper_views: Sequence[int],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_np = target.cpu().numpy().astype(np.int64)
    part_np = partslip_pred.cpu().numpy().astype(np.int64)
    cops_np = cops_pred.cpu().numpy().astype(np.int64)
    tile = rgbs.shape[1]
    label_h = 28
    title_h = 34
    gap = 8
    columns = ["RGB", "GT", "VG-AlignSeg", "PartSLIP2 adapter", "COPS adapter"]
    width = len(columns) * tile + (len(columns) + 1) * gap
    height = title_h + len(paper_views) * (tile + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()

    draw.text((gap, 8), f"Object {object_id}: 8-view part segmentation comparison", fill=(0, 0, 0), font=font)
    for col, label in enumerate(columns):
        x = gap + col * (tile + gap)
        draw.text((x + 4, title_h), label, fill=(20, 20, 20), font=small)

    for row, view_idx in enumerate(paper_views):
        y = title_h + label_h + row * (tile + label_h + gap)
        panels = [
            Image.fromarray(rgbs[view_idx], mode="RGB"),
            overlay_mask(rgbs[view_idx], target_np[view_idx]),
            crop_v4_prediction(v4_viz_path, view_idx, tile),
            overlay_mask(rgbs[view_idx], part_np[view_idx]),
            overlay_mask(rgbs[view_idx], cops_np[view_idx]),
        ]
        for col, panel in enumerate(panels):
            x = gap + col * (tile + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(220, 220, 220))
        draw.text((gap, y + tile + 4), view_names[view_idx].replace(".png", ""), fill=(60, 60, 60), font=small)
        if row + 1 < len(paper_views):
            for col, label in enumerate(columns):
                x = gap + col * (tile + gap)
                draw.text((x + 4, y + tile + 4), label, fill=(90, 90, 90), font=small)
    canvas.save(out_path)


def make_method_figure(
    out_path: Path,
    method_name: str,
    object_id: str,
    rgbs: np.ndarray,
    target: torch.Tensor,
    raw: np.ndarray,
    mapped: torch.Tensor,
    view_names: Sequence[str],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_np = target.cpu().numpy().astype(np.int64)
    mapped_np = mapped.cpu().numpy().astype(np.int64)
    tile = rgbs.shape[1]
    label_h = 26
    title_h = 34
    gap = 8
    columns = ["RGB", "GT overlay", "raw segments", f"{method_name} mapped"]
    width = len(columns) * tile + (len(columns) + 1) * gap
    height = title_h + len(view_names) * (tile + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    draw.text((gap, 8), f"{method_name} adapter visualization: object {object_id}", fill=(0, 0, 0), font=title_font)
    for col, label in enumerate(columns):
        x = gap + col * (tile + gap)
        draw.text((x + 4, title_h), label, fill=(30, 30, 30), font=font)
    for view_idx, name in enumerate(view_names):
        y = title_h + label_h + view_idx * (tile + label_h + gap)
        panels = [
            Image.fromarray(rgbs[view_idx], mode="RGB"),
            overlay_mask(rgbs[view_idx], target_np[view_idx]),
            mask_image(raw[view_idx]),
            overlay_mask(rgbs[view_idx], mapped_np[view_idx]),
        ]
        for col, panel in enumerate(panels):
            x = gap + col * (tile + gap)
            canvas.paste(panel, (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(220, 220, 220))
        draw.text((gap, y + tile + 4), name.replace(".png", ""), fill=(70, 70, 70), font=font)
    canvas.save(out_path)


def write_native_status(args: argparse.Namespace, out_dir: Path) -> None:
    statuses = {
        "partslip2": {
            "repository": "https://github.com/zyc00/PartSLIP2",
            "local_path": str(args.partslip2_repo),
            "commit": git_commit(args.partslip2_repo),
            "native_status": "not_directly_runnable_on_vg_alignseg_2d_only",
            "reason": (
                "PartSLIP2/PartSLIP++ expects 3D point-cloud/PartNetE-style inputs, "
                "category checkpoints, GLIP/SAM dependencies, and 3D-to-2D projection assets. "
                "VG-AlignSeg final test data provides object-centric 8-view RGBA images and 2D part masks."
            ),
            "reported_adapter": "PartSLIP2 2D proposal adapter; not a native PartSLIP2 result.",
        },
        "cops": {
            "repository": "https://github.com/marco-garosi/COPS",
            "local_path": str(args.cops_repo),
            "commit": git_commit(args.cops_repo),
            "native_status": "not_directly_runnable_on_vg_alignseg_2d_only",
            "reason": (
                "COPS expects 3D point clouds/meshes plus point-cloud feature aggregation. "
                "VG-AlignSeg final test data provides 2D rendered views and masks, not raw 3D point clouds."
            ),
            "reported_adapter": "COPS 2D cross-view clustering adapter; not a native COPS result.",
        },
    }
    for name, status in statuses.items():
        path = out_dir / name / "native_status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(status, indent=2), encoding="utf-8")


def add_metrics(row: dict, pixel_errors: int, pixel_count: int, mean_iou_sum: float, eval_batches: int) -> tuple[int, int, float]:
    pixel_errors += int(row["pixel_errors"])
    pixel_count += int(row["pixel_count"])
    mean_iou_sum += float(row["mean_iou"])
    return pixel_errors, pixel_count, mean_iou_sum


def summarize_method(
    method_name: str,
    protocol: str,
    per_object: list[dict],
    pixel_errors: int,
    pixel_count: int,
    mean_iou_sum: float,
    args: argparse.Namespace,
) -> dict:
    per_category = summarize_groups(per_object, "category")
    per_granularity = summarize_groups(per_object, "granularity")
    table_metrics = {
        "iou_object_category": mean([row["mean_iou"] for row in per_category.values()]),
        "iou_granularity": mean([row["mean_iou"] for row in per_granularity.values()]),
        "iou_part": mean_iou_sum / max(1, len(per_object)),
        "cross_view_consistency_acc": 1.0 - pixel_errors / max(1, pixel_count),
    }
    return {
        "baseline": method_name,
        "protocol": protocol,
        "metric_note": (
            "Adapter predictions are class-agnostic segment ids. Hungarian matching assigns segment ids "
            "to VG-AlignSeg actor ids for evaluation and mapped visualization only; GT masks are not copied."
        ),
        "split": "test",
        "shuffle_split": args.shuffle_split,
        "split_seed": args.split_seed,
        "train_count": args.train_count,
        "val_count": args.val_count,
        "test_count": len(per_object),
        "image_size": args.image_size,
        "views": args.views,
        "eval_mean_iou": table_metrics["iou_part"],
        "eval_pixel_accuracy": table_metrics["cross_view_consistency_acc"],
        "eval_pixel_errors": pixel_errors,
        "eval_pixel_count": pixel_count,
        "eval_batches": len(per_object),
        "table_metrics": table_metrics,
        "per_category": per_category,
        "per_granularity": per_granularity,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    category_map = load_category_map(args.category_map)
    dataset = ResultMultiViewDataset(
        args.data_root,
        split="test",
        views=args.views,
        image_size=args.image_size,
        train_count=optional_count(args.train_count),
        val_count=optional_count(args.val_count),
        test_count=optional_count(args.test_count),
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_native_status(args, args.output_dir)
    dataset.write_manifest(args.output_dir / "test_manifest.json")

    paper_views = parse_paper_views(args.paper_views, args.views)
    per_object: dict[str, list[dict]] = {"partslip2": [], "cops": []}
    pixel_errors = defaultdict(int)
    pixel_count = defaultdict(int)
    mean_iou_sum = defaultdict(float)
    raw_dirs = {
        "partslip2": args.output_dir / "partslip2" / "raw_outputs",
        "cops": args.output_dir / "cops" / "raw_outputs",
    }
    method_viz_dirs = {
        "partslip2": args.output_dir / "partslip2" / "visualizations",
        "cops": args.output_dir / "cops" / "visualizations",
    }
    paper_dir = args.output_dir / "paper_figures"

    total = len(dataset) if args.max_objects is None else min(len(dataset), args.max_objects)
    explicit_viz_indices = parse_viz_indices(args.viz_indices, total)
    if explicit_viz_indices is not None:
        viz_indices = explicit_viz_indices
    elif args.viz_random:
        rng = np.random.default_rng(args.viz_seed)
        viz_indices = set(int(x) for x in rng.choice(total, size=min(args.viz_samples, total), replace=False).tolist())
    else:
        viz_indices = set(range(min(args.viz_samples, total)))
    viz_indices_sorted = sorted(viz_indices)
    for idx in range(total):
        item = dataset[idx]
        obj = dataset.objects[idx]
        object_id = str(item["object_id"])
        view_names = list(item["view_names"])
        if args.log_every > 0 and (idx == 0 or (idx + 1) % args.log_every == 0):
            print(f"eval progress: {idx + 1}/{total} ({object_id})", flush=True)
        rgbs, fg = load_object_rgb_alpha(obj.root, view_names, args.image_size)
        target = item["masks"].to(torch.long)
        class_ids = [int(actor_id) - 1 for actor_id in item["actor_ids"]]
        k = max(1, len(class_ids))

        raw_part = partslip2_2d_adapter(rgbs, fg, k, args.seed + idx * 17, args.max_cluster_pixels)
        raw_cops = cops_2d_adapter(rgbs, fg, k, args.seed + idx * 23, args.max_cluster_pixels)
        mapped_part, metrics_part = oracle_match_prediction(torch.from_numpy(raw_part), target, class_ids)
        mapped_cops, metrics_cops = oracle_match_prediction(torch.from_numpy(raw_cops), target, class_ids)

        rows = {
            "partslip2": metrics_part,
            "cops": metrics_cops,
        }
        for name, metrics in rows.items():
            row = {
                "index": idx,
                "object_id": object_id,
                "category": category_map.get(object_id, "Unknown"),
                "actor_count": len(class_ids),
                "granularity": actor_count_bucket(len(class_ids)),
                "mean_iou": float(metrics["mean_iou"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
                "pixel_errors": int(metrics["pixel_errors"]),
                "pixel_count": int(metrics["pixel_count"]),
                "actors": int(metrics["actors"]),
                "pred_segments": int(metrics["pred_segments"]),
                "matched_pairs": metrics["matched_pairs"],
                "per_class": metrics["per_class"],
            }
            per_object[name].append(row)
            pixel_errors[name], pixel_count[name], mean_iou_sum[name] = add_metrics(
                row, pixel_errors[name], pixel_count[name], mean_iou_sum[name], len(per_object[name])
            )

        if idx in viz_indices:
            object_tag = f"{idx:04d}_{object_id}"
            save_mask_set(raw_dirs["partslip2"], object_tag, raw_part, mapped_part, view_names)
            save_mask_set(raw_dirs["cops"], object_tag, raw_cops, mapped_cops, view_names)
            make_method_figure(
                method_viz_dirs["partslip2"] / f"{object_tag}.png",
                "PartSLIP2",
                object_id,
                rgbs,
                target,
                raw_part,
                mapped_part,
                view_names,
            )
            make_method_figure(
                method_viz_dirs["cops"] / f"{object_tag}.png",
                "COPS",
                object_id,
                rgbs,
                target,
                raw_cops,
                mapped_cops,
                view_names,
            )
            v4_viz = find_v4_viz(args.v4_viz_dir, idx, object_id)
            make_paper_figure(
                paper_dir / f"{object_tag}.png",
                object_id,
                rgbs,
                target,
                v4_viz,
                mapped_part,
                mapped_cops,
                view_names,
                paper_views,
            )

    summaries = {
        "partslip2": summarize_method(
            "PartSLIP2 2D adapter",
            "per-view foreground RGB+XY proposal clustering, Hungarian matched to actor ids",
            per_object["partslip2"],
            pixel_errors["partslip2"],
            pixel_count["partslip2"],
            mean_iou_sum["partslip2"],
            args,
        ),
        "cops": summarize_method(
            "COPS 2D adapter",
            "global cross-view foreground RGB+XY+view clustering, Hungarian matched to actor ids",
            per_object["cops"],
            pixel_errors["cops"],
            pixel_count["cops"],
            mean_iou_sum["cops"],
            args,
        ),
    }
    for name, summary in summaries.items():
        method_dir = args.output_dir / name
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (method_dir / "per_object.json").write_text(json.dumps(per_object[name], indent=2), encoding="utf-8")

    combined = {
        "note": (
            "These are transparent 2D VG-AlignSeg adapters for visualization/comparison. "
            "The native PartSLIP2 and COPS repositories are 3D part segmentation methods and are not directly "
            "runnable on the 2D-only VG-AlignSeg final test split."
        ),
        "output_dir": str(args.output_dir),
        "paper_figures": str(paper_dir),
        "viz_random": args.viz_random,
        "viz_seed": args.viz_seed,
        "viz_indices_arg": args.viz_indices,
        "viz_indices": viz_indices_sorted,
        "viz_samples": len(viz_indices_sorted),
        "methods": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    main()
