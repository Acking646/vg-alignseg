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
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denom = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denom + 1.0)).mean()


def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, object]:
    pred = logits.argmax(dim=2)
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
