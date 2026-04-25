import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.shapenetpart import (
    DEFAULT_PART_COLORS,
    SEG_CLASSES,
    ShapeNetPartPointDataset,
    render_multiview_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one real ShapeNetPart sample")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/home/lyx/datasets/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--category", type=str, default="Chair")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(REPO_ROOT, "docs", "assets", "shapenetpart_real_chair_sample.png"),
    )
    parser.add_argument(
        "--npz-output",
        type=str,
        default=os.path.join(REPO_ROOT, "docs", "assets", "shapenetpart_real_chair_sample.npz"),
    )
    return parser.parse_args()


def colorize_mask(mask: np.ndarray, max_parts: int) -> np.ndarray:
    colors = DEFAULT_PART_COLORS[: max_parts + 1]
    mask = np.clip(mask, 0, max_parts)
    return colors[mask]


def main() -> None:
    args = parse_args()
    dataset = ShapeNetPartPointDataset(
        root=args.dataset_root,
        split=args.split,
        category=args.category,
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found for category={args.category}, split={args.split}")
    sample = dataset[args.index % len(dataset)]
    rendered = render_multiview_sample(sample=sample, image_size=args.image_size)

    images = rendered["images"]
    masks = rendered["masks"]
    view_names = rendered["view_names"]
    num_parts = len(SEG_CLASSES[args.category])
    masks_rgb = np.stack([colorize_mask(mask, max_parts=num_parts) for mask in masks], axis=0)

    fig, axes = plt.subplots(len(view_names), 2, figsize=(7.5, 12))
    fig.patch.set_facecolor("white")
    col_titles = ["Rendered Input", "Target Part Mask"]

    for row_idx, view_name in enumerate(view_names):
        axes[row_idx, 0].imshow(images[row_idx])
        axes[row_idx, 1].imshow(masks_rgb[row_idx])

        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=13, weight="bold", pad=10)
        axes[row_idx, 0].set_ylabel(f"View {row_idx + 1}\n{view_name}", fontsize=11, rotation=0, labelpad=28, va="center")

    fig.text(0.08, 0.975, f"ShapeNetPart Real Sample: {sample.category_name} / {sample.object_id}", fontsize=16, weight="bold")
    fig.text(
        0.08,
        0.948,
        "This figure is rendered from the real ShapeNetPart point sample using four canonical viewpoints.",
        fontsize=10,
    )

    legend_start_x = 0.54
    part_names = [f"Part {idx}" for idx in range(num_parts)]
    if sample.category_name == "Chair":
        part_names = ["Back", "Seat", "Leg", "Arm"]

    for idx, name in enumerate(part_names):
        color = DEFAULT_PART_COLORS[idx + 1] / 255.0
        rect = plt.Rectangle((legend_start_x + idx * 0.095, 0.944), 0.018, 0.014, color=color, transform=fig.transFigure, clip_on=False)
        fig.add_artist(rect)
        fig.text(legend_start_x + idx * 0.095 + 0.022, 0.951, name, fontsize=9, va="center")

    plt.tight_layout(rect=[0.05, 0.04, 0.98, 0.93])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=180)
    plt.close(fig)

    np.savez(
        args.npz_output,
        images=images,
        masks=masks,
        category=sample.category_name,
        object_id=sample.object_id,
        synset_id=sample.synset_id,
        view_names=np.array(view_names),
    )

    print(args.output)
    print(args.npz_output)


if __name__ == "__main__":
    main()
