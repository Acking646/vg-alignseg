import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
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
from models import VGAlignSegV2  # noqa: E402
from scripts.eval_utils import (  # noqa: E402
    class_weights,
    dice_loss,
    flatten_logits,
    flatten_target,
    save_visualization,
    segmentation_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VG-AlignSeg V2 on the full result 8-view dataset.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "v2_result")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs. Use 0 or a negative value to train forever.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=194)
    parser.add_argument("--eval-split", type=str, choices=("val", "test"), default="test")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.1)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.0)
    parser.add_argument("--consistency-start-step", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--refine-hidden-dim", type=int, default=128)
    parser.add_argument("--no-prototypes", action="store_true")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--viz-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    return parser.parse_args()


def setup_distributed(args: argparse.Namespace) -> tuple[bool, int, int, int, str]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() and args.device == "cuda" else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", local_rank))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if distributed and device == "cuda":
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def print_rank0(rank: int, *args, **kwargs) -> None:
    if is_rank0(rank):
        print(*args, **kwargs)


def barrier(distributed: bool) -> None:
    if distributed:
        dist.barrier()


def reduce_mean(value: float, device: str, distributed: bool) -> float:
    tensor = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return float(tensor.item())


def reduce_sum(value: float, device: str, distributed: bool) -> float:
    tensor = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(items: Iterable[Dict[str, object]]) -> Dict[str, object]:
    items = list(items)
    return {
        "images": torch.stack([item["images"] for item in items], dim=0),
        "masks": torch.stack([item["masks"] for item in items], dim=0),
        "object_id": [item["object_id"] for item in items],
        "view_names": [item["view_names"] for item in items],
        "actor_ids": [item["actor_ids"] for item in items],
    }


def trainable_state_dict(model: VGAlignSegV2) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")}


def load_trainable_state(model: VGAlignSegV2, checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("backbone.")]
    if missing or unexpected:
        print(f"Loaded with missing={missing}, unexpected={unexpected}", flush=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def save_checkpoint(path: Path, model: VGAlignSegV2, optimizer: torch.optim.Optimizer, step: int, args, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "args": vars(args),
            "meta": meta,
        },
        path,
    )


def cached_backbone_outputs(
    model: VGAlignSegV2,
    images: torch.Tensor,
    object_ids: list[str],
    cache_dir: Path,
    device: str,
) -> Dict[str, torch.Tensor]:
    if len(object_ids) != 1:
        # Keep the first full-dataset version simple and robust.
        raise ValueError("cached_backbone_outputs currently expects batch_size=1")
    object_id = object_ids[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{object_id}.pt"
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        model.backbone.to(device)
        model.backbone.eval()
        with torch.no_grad():
            outputs = model.backbone(images.to(device))
        cached = {
            key: value.detach().cpu().to(torch.float16 if key == "token_grid" else torch.float32)
            for key, value in outputs.items()
            if key in {"token_grid", "point_grid", "point_conf_grid"}
        }
        torch.save(cached, cache_path)
    return {
        key: value.to(device=device, dtype=torch.float32, non_blocking=True)
        for key, value in cached.items()
    }


def consistency_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    final = outputs["final_logits"]
    propagated = outputs["propagated_logits_lowres"]
    bsz, n_views, num_classes, height, width = final.shape
    grid_h, grid_w = propagated.shape[2:4]
    final_low = F.adaptive_avg_pool2d(final.reshape(bsz * n_views, num_classes, height, width), (grid_h, grid_w))
    final_low = final_low.reshape(bsz, n_views, num_classes, grid_h, grid_w).permute(0, 1, 3, 4, 2)
    return F.kl_div(F.log_softmax(final_low, dim=-1), F.softmax(propagated.detach(), dim=-1), reduction="batchmean")


def evaluate_subset(
    model: VGAlignSegV2,
    loader: DataLoader,
    cache_dir: Path,
    device: str,
    num_classes: int,
    max_batches: int | None,
) -> Dict[str, float]:
    model.eval()
    totals = {"pixel_errors": 0, "pixel_count": 0, "mean_iou_sum": 0.0, "batches": 0}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["images"].to(device)
            target = batch["masks"].to(device=device, dtype=torch.long)
            cached = cached_backbone_outputs(model, images, batch["object_id"], cache_dir, device)
            outputs = model.forward_from_backbone_outputs(images, cached, output_size=images.shape[-2:])
            metrics = segmentation_metrics(outputs["final_logits"], target, num_classes)
            totals["pixel_errors"] += metrics["pixel_errors"]
            totals["pixel_count"] += int(target.numel())
            totals["mean_iou_sum"] += metrics["mean_iou"]
            totals["batches"] += 1
    if totals["batches"] == 0:
        return {"eval_pixel_accuracy": 0.0, "eval_mean_iou": 0.0, "eval_pixel_errors": 0}
    return {
        "eval_pixel_accuracy": 1.0 - totals["pixel_errors"] / max(1, totals["pixel_count"]),
        "eval_mean_iou": totals["mean_iou_sum"] / totals["batches"],
        "eval_pixel_errors": totals["pixel_errors"],
        "eval_batches": totals["batches"],
    }


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    seed_everything(args.seed + rank)
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for the first full-dataset training script.")

    if is_rank0(rank):
        args.output_dir.mkdir(parents=True, exist_ok=True)
    barrier(distributed)

    train_set = ResultMultiViewDataset(
        args.data_root,
        split="train",
        views=args.views,
        image_size=args.image_size,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        max_samples=args.max_train_samples,
    )
    eval_set = ResultMultiViewDataset(
        args.data_root,
        split=args.eval_split,
        views=args.views,
        image_size=args.image_size,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        max_samples=args.max_val_samples,
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
        "num_classes": train_set.num_classes,
        "classes": train_set.classes,
        "train_samples": len(train_set),
        "eval_split": args.eval_split,
        "eval_samples": len(eval_set),
        "views": args.views,
        "image_size": args.image_size,
        "distributed": distributed,
        "world_size": world_size,
    }
    if is_rank0(rank):
        (args.output_dir / "run_config.json").write_text(
            json.dumps({"args": vars(args), "meta": meta}, indent=2, default=str)
        )
        train_set.write_manifest(args.output_dir / "train_manifest.json")
        eval_set.write_manifest(args.output_dir / f"{args.eval_split}_manifest.json")
    barrier(distributed)

    base_model = VGAlignSegV2(
        num_classes=train_set.num_classes,
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        min_confidence=args.min_confidence,
        refine_hidden_dim=args.refine_hidden_dim,
        use_prototypes=not args.no_prototypes,
    ).to(device)
    resume_checkpoint: dict = {}
    if args.resume is not None:
        resume_checkpoint = load_trainable_state(base_model, args.resume, device)
        print_rank0(rank, f"Resumed from {args.resume}", flush=True)

    base_model.backbone.eval()
    params = [p for name, p in base_model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if resume_checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        move_optimizer_state_to_device(optimizer, device)
        print_rank0(rank, "Resumed optimizer state.", flush=True)
    train_model = (
        DistributedDataParallel(
            base_model,
            device_ids=[local_rank] if device.startswith("cuda") else None,
            output_device=local_rank if device.startswith("cuda") else None,
            find_unused_parameters=args.no_prototypes,
        )
        if distributed
        else base_model
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    global_step = int(resume_checkpoint.get("step", 0))
    best_val_miou = -1.0

    print_rank0(
        rank,
        f"Training VG-AlignSeg V2 on result dataset: train={len(train_set)} "
        f"{args.eval_split}={len(eval_set)} classes={train_set.num_classes} cache={args.cache_dir} "
        f"world_size={world_size}",
        flush=True,
    )
    if args.eval_split == "test" and args.test_count is not None and len(eval_set) < args.test_count:
        print_rank0(
            rank,
            f"WARNING: requested test_count={args.test_count}, but only {len(eval_set)} "
            "complete 8-view test samples are available after the train split.",
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
            target = batch["masks"].to(device=device, dtype=torch.long, non_blocking=True)
            cached = cached_backbone_outputs(base_model, images, batch["object_id"], args.cache_dir, device)

            optimizer.zero_grad(set_to_none=True)
            outputs = train_model(images, cached, output_size=images.shape[-2:])
            final_flat = flatten_logits(outputs["final_logits"])
            coarse_flat = flatten_logits(outputs["coarse_logits"])
            target_flat = flatten_target(target)
            weights = class_weights(target.cpu(), train_set.num_classes).to(device)

            final_ce = F.cross_entropy(final_flat, target_flat, weight=weights)
            coarse_ce = F.cross_entropy(coarse_flat, target_flat, weight=weights)
            final_dice = dice_loss(final_flat, target_flat)
            cons = consistency_loss(outputs)
            cons_weight = args.consistency_loss_weight if global_step >= args.consistency_start_step else 0.0
            loss = final_ce + args.coarse_loss_weight * coarse_ce + args.dice_loss_weight * final_dice + cons_weight * cons
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            if global_step == 1 or global_step % args.log_every == 0:
                with torch.no_grad():
                    metrics = segmentation_metrics(outputs["final_logits"].detach(), target, train_set.num_classes)
                pixel_errors = reduce_sum(metrics["pixel_errors"], device, distributed)
                pixel_count = reduce_sum(target.numel(), device, distributed)
                row = {
                    "step": global_step,
                    "phase": "train",
                    "epoch": epoch,
                    "object_id": batch["object_id"][0],
                    "loss": reduce_mean(loss.item(), device, distributed),
                    "final_ce": reduce_mean(final_ce.item(), device, distributed),
                    "coarse_ce": reduce_mean(coarse_ce.item(), device, distributed),
                    "dice": reduce_mean(final_dice.item(), device, distributed),
                    "consistency": reduce_mean(cons.item(), device, distributed),
                    "consistency_weight": cons_weight,
                    "pixel_accuracy": 1.0 - pixel_errors / max(1.0, pixel_count),
                    "pixel_errors": int(pixel_errors),
                    "mean_iou": reduce_mean(metrics["mean_iou"], device, distributed),
                    "mean_dice": reduce_mean(metrics["mean_dice"], device, distributed),
                }
                if is_rank0(rank):
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
                    print(
                        f"step={global_step:06d} epoch={epoch:03d} "
                        f"loss={row['loss']:.4f} ce={row['final_ce']:.4f} dice={row['dice']:.4f} "
                        f"miou={row['mean_iou']:.4f} acc={row['pixel_accuracy']:.4f} "
                        f"errors={row['pixel_errors']}",
                        flush=True,
                    )

            if is_rank0(rank) and (global_step == 1 or global_step % args.viz_every == 0):
                save_visualization(
                    args.output_dir / f"train_step_{global_step:06d}.png",
                    images.detach(),
                    target.detach(),
                    outputs["final_logits"].detach(),
                    batch["view_names"][0],
                )

            if global_step % args.save_every == 0:
                barrier(distributed)
                if is_rank0(rank):
                    eval_metrics = evaluate_subset(
                        base_model,
                        eval_loader,
                        args.cache_dir,
                        device,
                        train_set.num_classes,
                        max_batches=args.eval_max_batches,
                    )
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "step": global_step,
                                    "phase": args.eval_split,
                                    "epoch": epoch,
                                    **eval_metrics,
                                }
                            )
                            + "\n"
                        )
                    save_checkpoint(args.output_dir / "latest.pt", base_model, optimizer, global_step, args, meta)
                    if eval_metrics["eval_mean_iou"] > best_val_miou:
                        best_val_miou = eval_metrics["eval_mean_iou"]
                        save_checkpoint(args.output_dir / "best.pt", base_model, optimizer, global_step, args, meta)
                    print(
                        f"{args.eval_split}@{global_step}: miou={eval_metrics['eval_mean_iou']:.4f} "
                        f"acc={eval_metrics['eval_pixel_accuracy']:.4f} "
                        f"errors={eval_metrics['eval_pixel_errors']} batches={eval_metrics['eval_batches']}",
                        flush=True,
                    )
                barrier(distributed)
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
        print(f"Training complete at step={global_step}; saved latest checkpoint.", flush=True)
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
