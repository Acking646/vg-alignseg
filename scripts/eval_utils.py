from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from data.views_36845 import colorize


def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    bsz, n_views, num_classes, height, width = logits.shape
    return logits.reshape(bsz * n_views, num_classes, height, width)


def flatten_target(target: torch.Tensor) -> torch.Tensor:
    bsz, n_views, height, width = target.shape
    return target.reshape(bsz * n_views, height, width)


def mask_logits_to_actor_ids(logits: torch.Tensor, actor_ids: Sequence[Sequence[int]]) -> torch.Tensor:
    """Restrict class competition to background plus actor ids known for each object."""
    if not actor_ids:
        return logits
    if logits.dim() != 5:
        raise ValueError(f"Expected logits [B, V, C, H, W], got {tuple(logits.shape)}")
    bsz, _, num_classes, _, _ = logits.shape
    allowed = torch.zeros((bsz, num_classes), device=logits.device, dtype=torch.bool)
    allowed[:, 0] = True
    for batch_idx, ids in enumerate(actor_ids):
        for actor_id in ids:
            class_id = int(actor_id) - 1
            if 0 <= class_id < num_classes:
                allowed[batch_idx, class_id] = True
    return logits.masked_fill(~allowed[:, None, :, None, None], -1.0e4)


def class_weights(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(target.reshape(-1), minlength=num_classes).float()
    present = counts > 0
    weights = torch.ones(num_classes, dtype=torch.float32)
    if present.any():
        freq = counts[present] / counts[present].sum().clamp_min(1.0)
        present_weights = torch.rsqrt(freq.clamp_min(1e-6))
        present_weights = present_weights / present_weights.mean().clamp_min(1e-6)
        weights[present] = present_weights
    return weights.clamp(max=10.0)


def active_class_probs_and_targets(
    logits: torch.Tensor,
    target: torch.Tensor,
    include_background: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_classes = logits.shape[1]
    probs = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).to(probs.dtype)
    if not include_background and num_classes > 1:
        probs = probs[:, 1:]
        one_hot = one_hot[:, 1:]
    active = one_hot.sum(dim=(0, 2, 3)) > 0
    if active.any():
        probs = probs[:, active]
        one_hot = one_hot[:, active]
    return probs, one_hot


def dice_loss(logits: torch.Tensor, target: torch.Tensor, include_background: bool = False) -> torch.Tensor:
    probs, one_hot = active_class_probs_and_targets(logits, target, include_background=include_background)
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce).clamp(1e-6, 1.0)
    loss = ((1.0 - pt) ** gamma) * ce
    if weight is not None:
        alpha = weight.to(device=logits.device, dtype=loss.dtype)[target].clamp_min(1e-6)
        return (loss * alpha).sum() / alpha.sum().clamp_min(1e-6)
    return loss.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    include_background: bool = False,
) -> torch.Tensor:
    probs, one_hot = active_class_probs_and_targets(logits, target, include_background=include_background)
    dims = (0, 2, 3)
    tp = (probs * one_hot).sum(dim=dims)
    fp = (probs * (1.0 - one_hot)).sum(dim=dims)
    fn = ((1.0 - probs) * one_hot).sum(dim=dims)
    score = (tp + 1.0) / (tp + alpha * fp + beta * fn + 1.0)
    return 1.0 - score.mean()


def _soft_boundary(x: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    pad = kernel_size // 2
    dilated = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    include_background: bool = False,
    kernel_size: int = 3,
) -> torch.Tensor:
    probs, one_hot = active_class_probs_and_targets(logits, target, include_background=include_background)
    pred_boundary = _soft_boundary(probs, kernel_size=kernel_size)
    target_boundary = _soft_boundary(one_hot, kernel_size=kernel_size)
    intersection = (pred_boundary * target_boundary).sum(dim=(0, 2, 3))
    denom = pred_boundary.sum(dim=(0, 2, 3)) + target_boundary.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()


def segmentation_metrics_from_prediction(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, object]:
    errors = int((pred != target).sum().item())
    pixel_acc = float((pred == target).float().mean().item())

    per_class_iou = {}
    per_class_dice = {}
    for class_id in range(num_classes):
        pred_c = pred == class_id
        target_c = target == class_id
        union = (pred_c | target_c).sum()
        inter = (pred_c & target_c).sum()
        target_count = target_c.sum()
        pred_count = pred_c.sum()
        if union > 0:
            per_class_iou[str(class_id)] = float((inter.float() / union.float()).item())
            per_class_dice[str(class_id)] = float(((2 * inter).float() / (pred_count + target_count).float()).item())
        else:
            per_class_iou[str(class_id)] = None
            per_class_dice[str(class_id)] = None

    valid_ious = [v for v in per_class_iou.values() if v is not None]
    valid_dice = [v for v in per_class_dice.values() if v is not None]
    return {
        "pixel_accuracy": pixel_acc,
        "pixel_errors": errors,
        "mean_iou": float(np.mean(valid_ious)) if valid_ious else 0.0,
        "mean_dice": float(np.mean(valid_dice)) if valid_dice else 0.0,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
    }


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, object]:
    pred = logits.argmax(dim=2)
    return segmentation_metrics_from_prediction(pred, target, num_classes)


def cleanup_prediction(pred: torch.Tensor, min_area: int = 16, fill_holes: bool = True) -> torch.Tensor:
    if min_area <= 0 and not fill_holes:
        return pred
    from scipy import ndimage

    pred_np = pred.detach().cpu().numpy().astype(np.int64)
    cleaned = pred_np.copy()
    for flat_idx in np.ndindex(pred_np.shape[:-2]):
        mask = pred_np[flat_idx]
        out = cleaned[flat_idx]
        for class_id in np.unique(mask):
            if class_id == 0:
                continue
            binary = mask == class_id
            labeled, num = ndimage.label(binary)
            if min_area > 0:
                for comp_id in range(1, num + 1):
                    comp = labeled == comp_id
                    if int(comp.sum()) < min_area:
                        out[comp] = 0
            if fill_holes:
                binary = out == class_id
                filled = ndimage.binary_fill_holes(binary)
                holes = filled & ~binary
                if holes.any() and int(holes.sum()) <= max(min_area * 4, 64):
                    out[holes] = class_id
    return torch.from_numpy(cleaned).to(device=pred.device, dtype=pred.dtype)


def segmentation_metrics_with_cleanup(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    min_area: int = 16,
    fill_holes: bool = True,
) -> Dict[str, object]:
    pred = logits.argmax(dim=2)
    pred = cleanup_prediction(pred, min_area=min_area, fill_holes=fill_holes)
    return segmentation_metrics_from_prediction(pred, target, num_classes)


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
        labels = [name.replace(".png", ""), "GT", "prediction"]
        for col, panel in enumerate(panels):
            x = col * tile_w
            canvas.paste(panel, (x, y + label_h))
            draw.text((x + 6, y + 5), labels[col], fill=(0, 0, 0))
    canvas.save(output_path)
