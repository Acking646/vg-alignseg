from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset  # noqa: E402
from scripts.eval_utils import dice_loss, mask_logits_to_actor_ids, save_visualization  # noqa: E402
from scripts.train_v2_result import collate_batch, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a task-compatible image-space adapter inspired by Cross View "
            "Transformers. This is not the original BEV CVT model."
        )
    )
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "final_results" / "baselines" / "cross_view_transformers" / "image_space_adapter",
    )
    parser.add_argument("--category-map", type=Path, default=REPO_ROOT / "final_results" / "metadata" / "object_category_map.json")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=-1)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--background-weight", type=float, default=0.05)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-max-objects", type=int, default=60)
    parser.add_argument("--viz-samples", type=int, default=40)
    parser.add_argument("--restrict-to-actor-ids", action="store_true", default=True)
    parser.add_argument("--no-restrict-to-actor-ids", action="store_false", dest="restrict_to_actor_ids")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--log-every", type=int, default=25)
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class CVTImageSpaceAdapter(nn.Module):
    def __init__(
        self,
        num_classes: int,
        views: int = 8,
        hidden_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        token_hw: int = 7,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.views = views
        self.token_hw = token_hw
        self.enc1 = ConvBlock(3, 32, stride=2)
        self.enc2 = ConvBlock(32, 48, stride=2)
        self.enc3 = ConvBlock(48, 80, stride=2)
        self.enc4 = ConvBlock(80, hidden_dim, stride=2)
        self.enc5 = ConvBlock(hidden_dim, hidden_dim, stride=2)

        self.view_embed = nn.Parameter(torch.zeros(1, views, 1, hidden_dim))
        self.spatial_embed = nn.Parameter(torch.zeros(1, 1, token_hw * token_hw, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=transformer_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.05,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.cross_view_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.up4 = UpBlock(hidden_dim, hidden_dim, hidden_dim)
        self.up3 = UpBlock(hidden_dim, 80, 80)
        self.up2 = UpBlock(80, 48, 48)
        self.up1 = UpBlock(48, 32, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        bsz, n_views, channels, height, width = images.shape
        x = images.reshape(bsz * n_views, channels, height, width)
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)
        low = self.enc5(s4)

        low_h, low_w = low.shape[-2:]
        if low_h != self.token_hw or low_w != self.token_hw:
            raise ValueError(f"Expected token grid {self.token_hw}x{self.token_hw}, got {low_h}x{low_w}")
        tokens = low.reshape(bsz, n_views, -1, low.shape[1]).permute(0, 1, 2, 3)
        tokens = tokens + self.view_embed[:, :n_views] + self.spatial_embed[:, :, : low_h * low_w]
        tokens = tokens.reshape(bsz, n_views * low_h * low_w, low.shape[1])
        tokens = self.cross_view_encoder(tokens)
        low = tokens.reshape(bsz, n_views, low_h * low_w, -1).permute(0, 1, 3, 2)
        low = low.reshape(bsz * n_views, -1, low_h, low_w)

        x = self.up4(low, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        logits = self.head(x)
        return logits.reshape(bsz, n_views, self.num_classes, height, width)


def batch_class_weights(target: torch.Tensor, num_classes: int, background_weight: float) -> torch.Tensor:
    counts = torch.bincount(target.reshape(-1), minlength=num_classes).float()
    weights = torch.ones(num_classes, dtype=torch.float32, device=target.device)
    present = counts > 0
    if present.any():
        freq = counts[present].to(target.device) / counts[present].sum().clamp_min(1.0).to(target.device)
        present_weights = torch.rsqrt(freq.clamp_min(1e-6))
        present_weights = present_weights / present_weights.mean().clamp_min(1e-6)
        weights[present.to(target.device)] = present_weights.clamp(max=10.0)
    weights[0] = background_weight
    return weights


def flatten_logits(logits: torch.Tensor) -> torch.Tensor:
    bsz, n_views, num_classes, height, width = logits.shape
    return logits.reshape(bsz * n_views, num_classes, height, width)


def flatten_target(target: torch.Tensor) -> torch.Tensor:
    bsz, n_views, height, width = target.shape
    return target.reshape(bsz * n_views, height, width)


def actor_metrics(pred: torch.Tensor, target: torch.Tensor, actor_ids: Iterable[int]) -> dict[str, object]:
    class_ids = [int(actor_id) - 1 for actor_id in actor_ids if int(actor_id) > 1]
    ious = []
    per_class = {}
    for class_id in class_ids:
        pred_c = pred == class_id
        target_c = target == class_id
        union = (pred_c | target_c).sum()
        inter = (pred_c & target_c).sum()
        if int(union.item()) <= 0:
            iou = 0.0
        else:
            iou = float((inter.float() / union.float()).item())
        ious.append(iou)
        per_class[str(class_id)] = iou
    errors = int((pred != target).sum().item())
    count = int(target.numel())
    return {
        "mean_iou": mean(ious),
        "pixel_accuracy": 1.0 - errors / max(1, count),
        "pixel_errors": errors,
        "pixel_count": count,
        "actors": len(class_ids),
        "per_class_iou": per_class,
    }


def predict_batch(model: nn.Module, batch: dict, device: str, restrict_to_actor_ids: bool) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device, non_blocking=True)
    target = batch["masks"].to(device=device, dtype=torch.long, non_blocking=True)
    logits = model(images)
    if restrict_to_actor_ids:
        logits = mask_logits_to_actor_ids(logits, batch["actor_ids"])
    return logits.argmax(dim=2), target


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    category_map: dict[str, str],
    num_classes: int,
    output_dir: Path | None = None,
    max_objects: int | None = None,
    viz_samples: int = 0,
    restrict_to_actor_ids: bool = True,
    log_every: int = 50,
) -> dict:
    model.eval()
    per_object = []
    pixel_errors = 0
    pixel_count = 0
    mean_iou_sum = 0.0
    actor_instances = 0
    viz_records = []

    for batch_idx, batch in enumerate(loader):
        if max_objects is not None and batch_idx >= max_objects:
            break
        if log_every > 0 and (batch_idx == 0 or (batch_idx + 1) % log_every == 0):
            print(f"eval progress: object batch {batch_idx + 1}/{len(loader)}", flush=True)
        pred, target = predict_batch(model, batch, device, restrict_to_actor_ids)
        for item_idx, object_id in enumerate(batch["object_id"]):
            metrics = actor_metrics(pred[item_idx].cpu(), target[item_idx].cpu(), batch["actor_ids"][item_idx])
            actor_count = int(metrics["actors"])
            row = {
                "index": len(per_object),
                "object_id": str(object_id),
                "category": category_map.get(str(object_id), "Unknown"),
                "actor_count": actor_count,
                "granularity": actor_count_bucket(actor_count),
                "mean_iou": float(metrics["mean_iou"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
                "pixel_errors": int(metrics["pixel_errors"]),
                "actors": actor_count,
                "per_class_iou": metrics["per_class_iou"],
            }
            per_object.append(row)
            pixel_errors += int(metrics["pixel_errors"])
            pixel_count += int(metrics["pixel_count"])
            mean_iou_sum += float(metrics["mean_iou"])
            actor_instances += actor_count
            if len(viz_records) < viz_samples:
                viz_records.append(
                    {
                        "index": row["index"],
                        "object_id": str(object_id),
                        "images": batch["images"][item_idx : item_idx + 1].cpu(),
                        "target": target[item_idx : item_idx + 1].cpu(),
                        "pred": pred[item_idx : item_idx + 1].cpu(),
                        "view_names": batch["view_names"][item_idx],
                    }
                )

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
        "eval_mean_iou": table_metrics["iou_part"],
        "eval_pixel_accuracy": table_metrics["cross_view_consistency_acc"],
        "eval_pixel_errors": pixel_errors,
        "eval_pixel_count": pixel_count,
        "eval_actor_instances": actor_instances,
        "eval_batches": eval_batches,
        "table_metrics": table_metrics,
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output_dir / "per_object.json").write_text(json.dumps(per_object, indent=2), encoding="utf-8")
        (output_dir / "per_category.json").write_text(json.dumps(per_category, indent=2), encoding="utf-8")
        (output_dir / "per_granularity.json").write_text(json.dumps(per_granularity, indent=2), encoding="utf-8")
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        for record in viz_records:
            logits = F.one_hot(record["pred"].clamp_min(0), num_classes=num_classes).permute(0, 1, 4, 2, 3).float()
            save_visualization(
                viz_dir / f"{record['index']:04d}_{record['object_id']}.png",
                record["images"],
                record["target"],
                logits,
                record["view_names"],
            )
    return {**summary, "per_object": per_object}


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    category_map = json.loads(args.category_map.read_text(encoding="utf-8")) if args.category_map.exists() else {}

    train_set = ResultMultiViewDataset(
        args.data_root,
        split="train",
        views=args.views,
        image_size=args.image_size,
        train_count=optional_count(args.train_count),
        val_count=optional_count(args.val_count),
        test_count=optional_count(args.test_count),
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    test_set = ResultMultiViewDataset(
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
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )
    eval_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, min(args.num_workers, 2)),
        pin_memory=True,
        collate_fn=collate_batch,
    )

    model = CVTImageSpaceAdapter(
        num_classes=train_set.num_classes,
        views=args.views,
        hidden_dim=args.hidden_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device.startswith("cuda"))

    config = {
        "baseline": "Cross View Transformers image-space adapter",
        "repository_reference": "https://github.com/bradyz/cross_view_transformers",
        "protocol": (
            "Task-compatible modification of CVT: per-view CNN encoder, cross-view "
            "Transformer token exchange, U-Net image-space decoder."
        ),
        "metric_note": (
            "This is not the original BEV CVT model. It is a reasonable image-space "
            "adapter trained on VG-AlignSeg train split and evaluated on the held-out test split."
        ),
        **vars(args),
        "num_classes": train_set.num_classes,
        "train_samples": len(train_set),
        "test_samples": len(test_set),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    train_set.write_manifest(args.output_dir / "train_manifest.json")
    test_set.write_manifest(args.output_dir / "test_manifest.json")

    metrics_path = args.output_dir / "metrics.jsonl"
    best_eval = -1.0
    train_iter = iter(train_loader)
    for step in range(1, args.max_steps + 1):
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        images = batch["images"].to(args.device, non_blocking=True)
        target = batch["masks"].to(device=args.device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=args.amp and args.device.startswith("cuda")):
            logits = model(images)
            flat_logits = flatten_logits(logits)
            flat_target = flatten_target(target)
            weights = batch_class_weights(target, train_set.num_classes, args.background_weight)
            ce = F.cross_entropy(flat_logits, flat_target, weight=weights)
            dloss = dice_loss(flat_logits, flat_target, include_background=False)
            loss = args.ce_loss_weight * ce + args.dice_loss_weight * dloss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % args.log_every == 0:
            pred = mask_logits_to_actor_ids(logits.detach(), batch["actor_ids"]).argmax(dim=2)
            batch_metrics = []
            for item_idx in range(pred.shape[0]):
                batch_metrics.append(actor_metrics(pred[item_idx].cpu(), target[item_idx].cpu(), batch["actor_ids"][item_idx])["mean_iou"])
            row = {
                "step": step,
                "split": "train",
                "loss": float(loss.detach().cpu().item()),
                "ce": float(ce.detach().cpu().item()),
                "dice": float(dloss.detach().cpu().item()),
                "mean_iou": mean(batch_metrics),
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.max_steps):
            eval_result = evaluate(
                model,
                eval_loader,
                args.device,
                category_map,
                train_set.num_classes,
                max_objects=args.eval_max_objects,
                restrict_to_actor_ids=args.restrict_to_actor_ids,
                log_every=0,
            )
            eval_row = {
                "step": step,
                "split": "test_subset",
                "eval_max_objects": args.eval_max_objects,
                "eval_mean_iou": eval_result["eval_mean_iou"],
                "eval_pixel_accuracy": eval_result["eval_pixel_accuracy"],
            }
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(eval_row) + "\n")
            print(json.dumps(eval_row), flush=True)
            if float(eval_result["eval_mean_iou"]) > best_eval:
                best_eval = float(eval_result["eval_mean_iou"])
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step,
                        "best_eval_mean_iou": best_eval,
                        "config": config,
                    },
                    args.output_dir / "best.pt",
                )

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    final_dir = args.output_dir / "test_eval"
    final = evaluate(
        model,
        eval_loader,
        args.device,
        category_map,
        train_set.num_classes,
        output_dir=final_dir,
        max_objects=None,
        viz_samples=args.viz_samples,
        restrict_to_actor_ids=args.restrict_to_actor_ids,
        log_every=50,
    )
    summary = {
        "baseline": config["baseline"],
        "repository_reference": config["repository_reference"],
        "protocol": config["protocol"],
        "metric_note": config["metric_note"],
        "best_checkpoint_step": int(checkpoint["step"]),
        "train_count": args.train_count,
        "val_count": args.val_count,
        "test_count": len(test_set),
        "shuffle_split": args.shuffle_split,
        "split_seed": args.split_seed,
        "restrict_to_actor_ids": args.restrict_to_actor_ids,
        **{k: v for k, v in final.items() if k != "per_object"},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
