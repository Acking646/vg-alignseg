import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models import VGAlignSegV1  # noqa: E402


DEFAULT_DATA_ROOT = REPO_ROOT / "36845_views"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "overfit_36845"


PALETTE = np.array(
    [
        [0, 0, 0],
        [230, 57, 70],
        [29, 53, 87],
        [69, 123, 157],
        [42, 157, 143],
        [233, 196, 106],
        [244, 162, 97],
        [131, 56, 236],
        [255, 0, 110],
        [58, 134, 255],
        [138, 201, 38],
        [255, 202, 58],
        [106, 76, 147],
        [25, 130, 196],
        [255, 127, 80],
        [45, 106, 79],
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit VG-AlignSeg V1 on the local 36845 8-view sample.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--views", type=int, default=4, help="Use the first N sorted views.")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.3)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--viz-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--decoder", choices=["lowres", "highres"], default="lowres")
    parser.add_argument("--highres-hidden-dim", type=int, default=128)
    parser.add_argument("--resume-heads", type=Path, default=None)
    parser.add_argument("--hard-pixel-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--pixel-memorizer",
        action="store_true",
        help="Add a learnable per-view pixel residual for strict single-sample overfit diagnostics.",
    )
    parser.add_argument("--pixel-memorizer-lr", type=float, default=0.1)
    parser.add_argument(
        "--mask-mode",
        choices=["actors", "chair-semantic"],
        default="actors",
        help="actors keeps each actor_* as a class; chair-semantic merges to background/back/seat/base.",
    )
    parser.add_argument(
        "--actor-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional actor ids to use as classes. Default: all actor_* directories.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sorted_view_paths(data_root: Path, views: int) -> List[Path]:
    paths = sorted((data_root / "color").glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No color views found under {data_root / 'color'}")
    if views > len(paths):
        raise ValueError(f"Requested {views} views, but only found {len(paths)}")
    return paths[:views]


def actor_dirs(data_root: Path, actor_ids: Sequence[int] | None) -> List[Path]:
    dirs = sorted((data_root / "part_mask").glob("actor_*"), key=lambda p: int(p.name.split("_")[1]))
    if actor_ids is not None:
        wanted = {int(x) for x in actor_ids}
        dirs = [p for p in dirs if int(p.name.split("_")[1]) in wanted]
    if not dirs:
        raise FileNotFoundError(f"No actor mask directories found under {data_root / 'part_mask'}")
    return dirs


def square_pad_image(img: Image.Image, fill: Tuple[int, ...]) -> Image.Image:
    width, height = img.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    canvas = Image.new(img.mode, (side, side), fill)
    canvas.paste(img, (left, top))
    return canvas


def load_rgb(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(background, img).convert("RGB")
    img = square_pad_image(img, (255, 255, 255))
    img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def load_actor_mask(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = square_pad_image(img, (0,))
    img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(img) > 0


def load_label_rgb(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = square_pad_image(img, (0, 0, 0))
    img = img.resize((image_size, image_size), Image.Resampling.NEAREST)
    return np.asarray(img)


def split_shell_back_seat(shell_mask: np.ndarray, label_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    shell_pixels = int(shell_mask.sum())
    if shell_pixels == 0:
        return np.zeros_like(shell_mask), np.zeros_like(shell_mask)

    flat_colors, counts = np.unique(label_rgb[shell_mask], axis=0, return_counts=True)
    min_component_pixels = max(20, int(0.05 * shell_pixels))

    components = []
    for color, count in zip(flat_colors, counts):
        if int(count) < min_component_pixels or tuple(int(x) for x in color) == (0, 0, 0):
            continue
        component_mask = shell_mask & (label_rgb == color).all(axis=-1)
        ys, _ = np.where(component_mask)
        if len(ys) == 0:
            continue
        components.append((float(ys.mean()), int(count), component_mask))

    if components:
        components.sort(key=lambda item: item[0])
        back_mask = components[0][2]
        seat_mask = shell_mask & ~back_mask
        return back_mask, seat_mask

    ys, _ = np.where(shell_mask)
    split_y = float(np.median(ys))
    back_mask = shell_mask.copy()
    back_mask[np.arange(shell_mask.shape[0])[:, None] > split_y] = False
    seat_mask = shell_mask & ~back_mask
    return back_mask, seat_mask


def load_actor_sample(
    data_root: Path,
    view_paths: Sequence[Path],
    image_size: int,
    selected_actor_ids: Sequence[int] | None,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    dirs = actor_dirs(data_root, selected_actor_ids)
    actor_names = [p.name for p in dirs]

    masks = np.zeros((len(view_paths), image_size, image_size), dtype=np.int64)
    for class_id, actor_dir in enumerate(dirs, start=1):
        for view_idx, view_path in enumerate(view_paths):
            mask_path = actor_dir / view_path.name
            if not mask_path.exists():
                continue
            actor_mask = load_actor_mask(mask_path, image_size)
            masks[view_idx][actor_mask] = class_id

    meta = {
        "actor_classes": {"0": "background", **{str(i + 1): name for i, name in enumerate(actor_names)}},
    }
    return torch.from_numpy(masks).unsqueeze(0), meta


def load_chair_semantic_sample(
    data_root: Path,
    view_paths: Sequence[Path],
    image_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    actor14_dir = data_root / "part_mask" / "actor_14"
    if not actor14_dir.is_dir():
        raise FileNotFoundError(f"Expected shell mask directory at {actor14_dir}")

    base_dirs = [p for p in actor_dirs(data_root, None) if p.name != "actor_14"]
    masks = np.zeros((len(view_paths), image_size, image_size), dtype=np.int64)

    for view_idx, view_path in enumerate(view_paths):
        base_mask = np.zeros((image_size, image_size), dtype=bool)
        for actor_dir in base_dirs:
            mask_path = actor_dir / view_path.name
            if mask_path.exists():
                base_mask |= load_actor_mask(mask_path, image_size)
        masks[view_idx][base_mask] = 3

        shell_path = actor14_dir / view_path.name
        if not shell_path.exists():
            continue
        shell_mask = load_actor_mask(shell_path, image_size)
        label_rgb = load_label_rgb(data_root / "label0" / view_path.name, image_size)
        back_mask, seat_mask = split_shell_back_seat(shell_mask, label_rgb)
        masks[view_idx][seat_mask] = 2
        masks[view_idx][back_mask] = 1

    meta = {
        "actor_classes": {
            "0": "background",
            "1": "back",
            "2": "seat",
            "3": "base",
        },
        "semantic_mapping": {
            "back": "top major label0 component inside actor_14",
            "seat": "remaining actor_14 shell pixels",
            "base": "all actor_* masks except actor_14",
        },
    }
    return torch.from_numpy(masks).unsqueeze(0), meta


def load_single_sample(
    data_root: Path,
    num_views: int,
    image_size: int,
    selected_actor_ids: Sequence[int] | None,
    mask_mode: str = "actors",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    view_paths = sorted_view_paths(data_root, num_views)

    images = torch.stack([load_rgb(path, image_size) for path in view_paths], dim=0)
    if mask_mode == "actors":
        masks, mask_meta = load_actor_sample(data_root, view_paths, image_size, selected_actor_ids)
    elif mask_mode == "chair-semantic":
        masks, mask_meta = load_chair_semantic_sample(data_root, view_paths, image_size)
    else:
        raise ValueError(f"Unsupported mask mode: {mask_mode}")

    meta = {
        "view_names": [p.name for p in view_paths],
        "image_size": image_size,
        "mask_mode": mask_mode,
        **mask_meta,
    }
    return images.unsqueeze(0), masks, meta


def class_weights(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(target.reshape(-1), minlength=num_classes).float()
    freq = counts / counts.sum().clamp_min(1.0)
    weights = torch.rsqrt(freq.clamp_min(1e-6))
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.clamp(max=10.0)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, include_background: bool = False) -> torch.Tensor:
    num_classes = logits.shape[1]
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).to(probs.dtype)

    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]

    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + 1.0) / (denom + 1.0)
    return 1.0 - dice.mean()


def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    bsz, n_views, num_classes, height, width = logits.shape
    return logits.reshape(bsz * n_views, num_classes, height, width)


def flatten_target(target: torch.Tensor) -> torch.Tensor:
    bsz, n_views, height, width = target.shape
    return target.reshape(bsz * n_views, height, width)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class HighResRefineHead(nn.Module):
    def __init__(self, token_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.token_proj = nn.Conv2d(token_dim, hidden_dim, kernel_size=1)
        self.logit_proj = nn.Conv2d(2 * num_classes, hidden_dim // 2, kernel_size=1)
        self.lowres_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_dim),
        )
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, hidden_dim // 2, kernel_size=5, padding=2),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim // 2),
            nn.SiLU(inplace=True),
        )
        self.fullres_fuse = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 2 + 5, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(
        self,
        token_grid: torch.Tensor,
        coarse_logits: torch.Tensor,
        propagated_logits: torch.Tensor,
        images: torch.Tensor,
        output_size: Tuple[int, int],
    ) -> torch.Tensor:
        bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
        _, _, _, _, num_classes = coarse_logits.shape
        height, width = output_size

        tokens = token_grid.reshape(bsz * n_views, grid_h, grid_w, token_dim).permute(0, 3, 1, 2).contiguous()
        coarse = coarse_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes).permute(0, 3, 1, 2).contiguous()
        propagated = (
            propagated_logits.reshape(bsz * n_views, grid_h, grid_w, num_classes)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        image_flat = images.reshape(bsz * n_views, 3, height, width)

        lowres = torch.cat([self.token_proj(tokens), self.logit_proj(torch.cat([coarse, propagated], dim=1))], dim=1)
        lowres = self.lowres_fuse(lowres)
        lowres = F.interpolate(lowres, size=output_size, mode="bilinear", align_corners=False)

        rgb_features = self.rgb_encoder(image_flat)
        yy = torch.linspace(-1.0, 1.0, height, device=images.device, dtype=images.dtype)
        xx = torch.linspace(-1.0, 1.0, width, device=images.device, dtype=images.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(bsz * n_views, -1, -1, -1)

        fullres = torch.cat([lowres, rgb_features, image_flat, coords], dim=1)
        logits = self.fullres_fuse(fullres)
        return logits.reshape(bsz, n_views, num_classes, height, width).contiguous()


class PixelMemorizer(nn.Module):
    def __init__(self, n_views: int, num_classes: int, image_size: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, n_views, num_classes, image_size, image_size))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[1:] != self.bias.shape[1:]:
            raise ValueError(f"PixelMemorizer shape {tuple(self.bias.shape)} does not match logits {tuple(logits.shape)}")
        return logits + self.bias


def forward_from_cached_backbone(
    model: VGAlignSegV1,
    cached: Dict[str, torch.Tensor],
    output_size: Tuple[int, int],
    images: torch.Tensor | None = None,
    highres_refine_head: HighResRefineHead | None = None,
    pixel_memorizer: PixelMemorizer | None = None,
):
    token_grid = cached["token_grid"]
    point_grid = cached["point_grid"]
    point_conf_grid = cached["point_conf_grid"]

    coarse_logits = model.coarse_head(token_grid)
    coarse_logits_fullres = model._upsample_lowres_logits(coarse_logits, output_size)

    bsz, n_views, grid_h, grid_w, token_dim = token_grid.shape
    _, _, _, _, num_classes = coarse_logits.shape
    num_tokens = grid_h * grid_w

    flat_tokens = token_grid.reshape(bsz, n_views, num_tokens, token_dim)
    flat_points = point_grid.reshape(bsz, n_views, num_tokens, 3)
    flat_conf = point_conf_grid.reshape(bsz, n_views, num_tokens)
    flat_logits = coarse_logits.reshape(bsz, n_views, num_tokens, num_classes)

    propagated_by_view = []
    template = torch.zeros_like(flat_logits[:, 0])

    for target_idx in range(n_views):
        target_tokens = flat_tokens[:, target_idx]
        target_points = flat_points[:, target_idx]
        target_conf = flat_conf[:, target_idx]

        messages = []
        for source_idx in range(n_views):
            if source_idx == target_idx:
                continue

            prune_outputs = model.geometry_pruner(
                target_points=target_points,
                source_points=flat_points[:, source_idx],
                target_conf=target_conf,
                source_conf=flat_conf[:, source_idx],
            )
            align_outputs = model.sparse_align(
                target_tokens=target_tokens,
                source_tokens=flat_tokens[:, source_idx],
                target_points=target_points,
                source_points=flat_points[:, source_idx],
                candidate_idx=prune_outputs["candidate_idx"],
                candidate_mask=prune_outputs["candidate_mask"],
            )
            messages.append(
                model.propagator(
                    source_logits=flat_logits[:, source_idx],
                    alignment_weights=align_outputs["weights"],
                    candidate_idx=prune_outputs["candidate_idx"],
                )
            )

        propagated_by_view.append(model.propagator.aggregate(messages, template))

    propagated_logits = torch.stack(propagated_by_view, dim=1)
    propagated_logits = propagated_logits.reshape(bsz, n_views, grid_h, grid_w, num_classes)
    propagated_logits_fullres = model._upsample_lowres_logits(propagated_logits, output_size)

    if highres_refine_head is None:
        refine_outputs = model.refine_head(
            token_grid=token_grid,
            coarse_logits=coarse_logits,
            propagated_logits=propagated_logits,
            output_size=output_size,
        )
        final_logits_lowres = refine_outputs["lowres_logits"]
        final_logits = refine_outputs["fullres_logits"]
    else:
        if images is None:
            raise ValueError("images are required when using the highres decoder")
        final_logits_lowres = coarse_logits
        final_logits = highres_refine_head(
            token_grid=token_grid,
            coarse_logits=coarse_logits,
            propagated_logits=propagated_logits,
            images=images,
            output_size=output_size,
        )

    if pixel_memorizer is not None:
        final_logits = pixel_memorizer(final_logits)

    return {
        "coarse_logits_lowres": coarse_logits,
        "coarse_logits": coarse_logits_fullres,
        "propagated_logits_lowres": propagated_logits,
        "propagated_logits": propagated_logits_fullres,
        "final_logits_lowres": final_logits_lowres,
        "final_logits": final_logits,
    }


def pixel_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=2)
    return float((pred == target).float().mean().item())


def pixel_errors(logits: torch.Tensor, target: torch.Tensor) -> int:
    pred = logits.argmax(dim=2)
    return int((pred != target).sum().item())


def hard_pixel_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    pred = logits.argmax(dim=1)
    wrong = pred != target
    if not wrong.any():
        return logits.sum() * 0.0
    per_pixel = F.cross_entropy(logits, target, weight=weight, reduction="none")
    return per_pixel[wrong].mean()


def mean_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    pred = logits.argmax(dim=2)
    scores = []
    for class_id in range(num_classes):
        pred_c = pred == class_id
        target_c = target == class_id
        union = (pred_c | target_c).sum()
        if union == 0:
            continue
        inter = (pred_c & target_c).sum()
        scores.append((inter.float() / union.float()).item())
    return float(np.mean(scores)) if scores else 0.0


def colorize(mask: np.ndarray) -> Image.Image:
    palette = PALETTE
    if mask.max(initial=0) >= len(palette):
        extra_count = int(mask.max()) + 1 - len(palette)
        rng = np.random.default_rng(123)
        extra = rng.integers(20, 235, size=(extra_count, 3), dtype=np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    return Image.fromarray(palette[mask], mode="RGB")


def save_visualization(
    output_path: Path,
    images: torch.Tensor,
    target: torch.Tensor,
    logits: torch.Tensor,
    view_names: Sequence[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    images_np = (images[0].permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    target_np = target[0].detach().cpu().numpy().astype(np.int64)
    pred_np = logits[0].argmax(dim=1).detach().cpu().numpy().astype(np.int64)

    tile_w = images_np.shape[2]
    tile_h = images_np.shape[1]
    label_h = 24
    canvas = Image.new("RGB", (tile_w * 3, (tile_h + label_h) * len(view_names)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for row, name in enumerate(view_names):
        y = row * (tile_h + label_h)
        panels = [
            Image.fromarray(images_np[row], mode="RGB"),
            colorize(target_np[row]),
            colorize(pred_np[row]),
        ]
        labels = [name.replace(".png", ""), "GT actor mask", "prediction"]
        for col, panel in enumerate(panels):
            x = col * tile_w
            canvas.paste(panel, (x, y + label_h))
            draw.text((x + 6, y + 5), labels[col], fill=(0, 0, 0))

    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, target, meta = load_single_sample(
        args.data_root,
        args.views,
        args.image_size,
        args.actor_ids,
        mask_mode=args.mask_mode,
    )
    num_classes = len(meta["actor_classes"])
    images = images.to(device)
    target = target.to(device=device, dtype=torch.long)

    weights = class_weights(target.cpu(), num_classes).to(device)
    print(f"Loaded sample: images={tuple(images.shape)}, target={tuple(target.shape)}, classes={num_classes}")
    print("Classes:", json.dumps(meta["actor_classes"], ensure_ascii=False))

    model = VGAlignSegV1(
        num_classes=num_classes,
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        min_confidence=args.min_confidence,
    ).to(device)

    print("Caching frozen VGGT outputs...")
    model.eval()
    with torch.no_grad():
        cached = model.backbone(images)
        cached = {
            key: value.detach()
            for key, value in cached.items()
            if key in {"token_grid", "point_grid", "point_conf_grid"}
        }

    model.backbone.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    highres_refine_head = None
    if args.decoder == "highres":
        highres_refine_head = HighResRefineHead(
            token_dim=model.backbone.token_dim,
            num_classes=num_classes,
            hidden_dim=args.highres_hidden_dim,
        ).to(device)

    pixel_memorizer = None
    if args.pixel_memorizer:
        pixel_memorizer = PixelMemorizer(
            n_views=args.views,
            num_classes=num_classes,
            image_size=args.image_size,
        ).to(device)

    if args.resume_heads is not None:
        checkpoint = torch.load(args.resume_heads, map_location=device, weights_only=False)
        model.coarse_head.load_state_dict(checkpoint["coarse_head"])
        model.sparse_align.load_state_dict(checkpoint["sparse_align"])
        if highres_refine_head is not None and checkpoint.get("highres_refine_head") is not None:
            highres_refine_head.load_state_dict(checkpoint["highres_refine_head"])
        elif highres_refine_head is None:
            model.refine_head.load_state_dict(checkpoint["refine_head"])
        if pixel_memorizer is not None and checkpoint.get("pixel_memorizer") is not None:
            pixel_memorizer.load_state_dict(checkpoint["pixel_memorizer"])
        print(f"Resumed trainable heads from {args.resume_heads}")

    train_modules = [model.coarse_head, model.sparse_align]
    train_modules.append(highres_refine_head if highres_refine_head is not None else model.refine_head)
    if pixel_memorizer is not None:
        train_modules.append(pixel_memorizer)
    for module in train_modules:
        module.train()

    base_params = []
    for module in train_modules:
        if module is pixel_memorizer:
            continue
        base_params.extend([p for p in module.parameters() if p.requires_grad])

    optimizer_groups = []
    if base_params:
        optimizer_groups.append({"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay})
    if pixel_memorizer is not None:
        optimizer_groups.append(
            {
                "params": list(pixel_memorizer.parameters()),
                "lr": args.pixel_memorizer_lr,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    params = base_params + ([] if pixel_memorizer is None else list(pixel_memorizer.parameters()))

    meta_path = args.output_dir / "overfit_36845_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    best_miou = 0.0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = forward_from_cached_backbone(
            model,
            cached,
            (args.image_size, args.image_size),
            images=images,
            highres_refine_head=highres_refine_head,
            pixel_memorizer=pixel_memorizer,
        )

        flat_final = flatten_logits(outputs["final_logits"])
        flat_coarse = flatten_logits(outputs["coarse_logits"])
        flat_target = flatten_target(target)

        final_ce = F.cross_entropy(flat_final, flat_target, weight=weights)
        coarse_ce = F.cross_entropy(flat_coarse, flat_target, weight=weights)
        final_dice = dice_loss(flat_final, flat_target)
        loss = final_ce + args.coarse_loss_weight * coarse_ce + args.dice_loss_weight * final_dice
        hard_ce = hard_pixel_loss(flat_final, flat_target, weight=weights)
        loss = loss + args.hard_pixel_loss_weight * hard_ce

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            acc = pixel_accuracy(outputs["final_logits"], target)
            miou = mean_iou(outputs["final_logits"], target, num_classes)
            errors = pixel_errors(outputs["final_logits"], target)
            best_miou = max(best_miou, miou)
            print(
                f"step={step:04d} loss={loss.item():.4f} "
                f"final_ce={final_ce.item():.4f} dice={final_dice.item():.4f} hard_ce={hard_ce.item():.4f} "
                f"acc={acc:.4f} miou={miou:.4f} best_miou={best_miou:.4f} errors={errors}"
            )

        if step == 1 or step % args.viz_every == 0 or step == args.steps:
            save_visualization(
                args.output_dir / f"step_{step:04d}.png",
                images.detach(),
                target.detach(),
                outputs["final_logits"].detach(),
                meta["view_names"],
            )

    torch.save(
        {
            "coarse_head": model.coarse_head.state_dict(),
            "sparse_align": model.sparse_align.state_dict(),
            "refine_head": model.refine_head.state_dict(),
            "highres_refine_head": None if highres_refine_head is None else highres_refine_head.state_dict(),
            "pixel_memorizer": None if pixel_memorizer is None else pixel_memorizer.state_dict(),
            "meta": meta,
            "args": vars(args),
        },
        args.output_dir / "overfit_36845_heads.pt",
    )
    print(f"Saved outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
