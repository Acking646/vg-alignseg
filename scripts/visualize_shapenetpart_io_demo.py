import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


CANVAS_SIZE = 224

PART_ORDER = ["back", "seat", "leg", "arm"]
PART_INDEX = {name: idx for idx, name in enumerate(PART_ORDER)}

RGB_COLORS = {
    "background": (245, 245, 245),
    "outline": (35, 35, 35),
    "back": (108, 145, 204),
    "seat": (229, 166, 77),
    "leg": (88, 163, 109),
    "arm": (206, 102, 116),
}

MASK_COLORS = {
    0: (255, 255, 255),
    1: (66, 135, 245),
    2: (247, 179, 43),
    3: (74, 182, 93),
    4: (220, 87, 106),
}


def polygons_for_view(view_name: str) -> List[Tuple[str, List[Tuple[int, int]]]]:
    if view_name == "front":
        return [
            ("back", [(58, 28), (166, 28), (166, 96), (58, 96)]),
            ("seat", [(64, 100), (160, 100), (154, 136), (70, 136)]),
            ("arm", [(42, 82), (58, 82), (58, 126), (42, 126)]),
            ("arm", [(166, 82), (182, 82), (182, 126), (166, 126)]),
            ("leg", [(72, 136), (88, 136), (86, 198), (70, 198)]),
            ("leg", [(136, 136), (152, 136), (154, 198), (138, 198)]),
            ("leg", [(94, 136), (108, 136), (108, 192), (94, 192)]),
            ("leg", [(116, 136), (130, 136), (130, 192), (116, 192)]),
        ]
    if view_name == "left":
        return [
            ("back", [(50, 36), (86, 36), (86, 140), (50, 140)]),
            ("seat", [(82, 104), (170, 104), (156, 132), (70, 132)]),
            ("arm", [(78, 86), (104, 86), (104, 112), (78, 112)]),
            ("leg", [(82, 132), (98, 132), (94, 198), (78, 198)]),
            ("leg", [(136, 132), (152, 132), (150, 198), (134, 198)]),
        ]
    if view_name == "right":
        return [
            ("back", [(138, 36), (174, 36), (174, 140), (138, 140)]),
            ("seat", [(54, 104), (142, 104), (154, 132), (68, 132)]),
            ("arm", [(120, 86), (146, 86), (146, 112), (120, 112)]),
            ("leg", [(74, 132), (90, 132), (90, 198), (74, 198)]),
            ("leg", [(128, 132), (144, 132), (148, 198), (132, 198)]),
        ]
    if view_name == "oblique":
        return [
            ("back", [(68, 38), (142, 28), (158, 90), (88, 104)]),
            ("seat", [(74, 106), (156, 96), (170, 128), (88, 140)]),
            ("arm", [(146, 72), (166, 70), (174, 104), (154, 106)]),
            ("arm", [(60, 78), (78, 80), (88, 118), (68, 116)]),
            ("leg", [(92, 140), (108, 138), (104, 198), (88, 198)]),
            ("leg", [(132, 136), (148, 136), (150, 198), (134, 198)]),
            ("leg", [(74, 138), (88, 138), (86, 192), (72, 192)]),
        ]
    raise ValueError(f"Unknown view name: {view_name}")


def draw_panel(view_name: str, variant: str) -> Image.Image:
    bg = RGB_COLORS["background"] if variant == "rgb" else MASK_COLORS[0]
    image = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), bg)
    draw = ImageDraw.Draw(image)

    polygons = polygons_for_view(view_name)
    polygons = modify_polygons(polygons, view_name=view_name, variant=variant)

    for part_name, polygon in polygons:
        if variant == "rgb":
            fill = RGB_COLORS[part_name]
            outline = RGB_COLORS["outline"]
        else:
            fill = MASK_COLORS[PART_INDEX[part_name] + 1]
            outline = fill
        draw.polygon(polygon, fill=fill, outline=outline)

    if variant == "rgb":
        draw.line((34, 198, 190, 198), fill=(140, 140, 140), width=2)

    return image


def modify_polygons(
    polygons: List[Tuple[str, List[Tuple[int, int]]]],
    view_name: str,
    variant: str,
) -> List[Tuple[str, List[Tuple[int, int]]]]:
    modified = []
    for part_name, polygon in polygons:
        if variant == "coarse":
            if view_name == "front" and part_name == "arm" and polygon[0][0] < 100:
                shifted = [(x + 10, y + (4 if i > 1 else 0)) for i, (x, y) in enumerate(polygon)]
                modified.append((part_name, shifted))
                continue
            if view_name == "left" and part_name == "seat":
                shrunk = [(x + (-8 if idx in (0, 3) else 0), y) for idx, (x, y) in enumerate(polygon)]
                modified.append((part_name, shrunk))
                continue
            if view_name == "right" and part_name == "leg" and polygon[0][0] > 100:
                continue
            if view_name == "oblique" and part_name == "back":
                expanded = [(x + (8 if idx in (1, 2) else 0), y + (4 if idx in (2, 3) else 0)) for idx, (x, y) in enumerate(polygon)]
                modified.append((part_name, expanded))
                continue
        elif variant == "refined":
            if view_name == "oblique" and part_name == "arm" and polygon[0][0] > 100:
                refined = [(x - 4, y - 2) for x, y in polygon]
                modified.append((part_name, refined))
                continue
            if view_name == "left" and part_name == "seat":
                refined = [(x + (-2 if idx in (0, 3) else 0), y) for idx, (x, y) in enumerate(polygon)]
                modified.append((part_name, refined))
                continue
        modified.append((part_name, polygon))
    return modified


def build_figure(output_path: str) -> None:
    view_names = ["front", "left", "right", "oblique"]
    col_titles = ["Input RGB", "GT Part Mask", "Coarse Output", "Refined Output"]
    variants = ["rgb", "mask", "coarse", "refined"]

    fig, axes = plt.subplots(len(view_names), len(variants), figsize=(12, 11))
    fig.patch.set_facecolor("white")

    for row_idx, view_name in enumerate(view_names):
        for col_idx, variant in enumerate(variants):
            img = np.asarray(draw_panel(view_name, variant))
            ax = axes[row_idx, col_idx]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)

            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=13, weight="bold", pad=10)
            if col_idx == 0:
                ax.set_ylabel(
                    {
                        "front": "View 1\nFront",
                        "left": "View 2\nLeft",
                        "right": "View 3\nRight",
                        "oblique": "View 4\nOblique",
                    }[view_name],
                    fontsize=12,
                    rotation=0,
                    labelpad=34,
                    va="center",
                )

    legend_items = [
        ("Back", MASK_COLORS[1]),
        ("Seat", MASK_COLORS[2]),
        ("Leg", MASK_COLORS[3]),
        ("Arm", MASK_COLORS[4]),
    ]

    fig.text(0.08, 0.975, "VG-AlignSeg on ShapeNetPart-style Chair Sample", fontsize=18, weight="bold")
    fig.text(
        0.08,
        0.945,
        "Task visualization: 4-view input, per-view 2D part supervision, coarse prediction, and geometry-guided refined output",
        fontsize=11,
    )

    start_x = 0.73
    y = 0.95
    for idx, (label, color) in enumerate(legend_items):
        rect = plt.Rectangle((start_x + idx * 0.06, y - 0.016), 0.018, 0.018, color=np.array(color) / 255.0, transform=fig.transFigure, clip_on=False)
        fig.add_artist(rect)
        fig.text(start_x + idx * 0.06 + 0.022, y - 0.002, label, fontsize=10, va="center")

    fig.text(
        0.08,
        0.02,
        "This is a task-level illustration generated locally because ShapeNetPart is not yet downloaded in the workspace. "
        "Once the real dataset is connected, the same IO layout becomes: 4 RGB views -> 4 GT part masks -> model coarse/final masks.",
        fontsize=10,
    )

    plt.tight_layout(rect=[0.04, 0.05, 0.98, 0.92])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(repo_root, "docs", "assets", "shapenetpart_v1_io_demo.png")
    build_figure(output_path)
    print(output_path)


if __name__ == "__main__":
    main()
