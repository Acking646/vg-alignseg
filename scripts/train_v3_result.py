import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset  # noqa: E402
from models import VGAlignSegV3  # noqa: E402
from scripts.eval_utils import (  # noqa: E402
    boundary_loss,
    class_weights,
    cleanup_prediction,
    dice_loss,
    flatten_logits,
    flatten_target,
    focal_loss,
    mask_logits_to_actor_ids,
    save_visualization,
    segmentation_metrics,
    segmentation_metrics_from_prediction,
    segmentation_metrics_with_cleanup,
    tversky_loss,
)
from scripts.train_v2_result import (  # noqa: E402
    barrier,
    cached_backbone_outputs,
    cleanup_distributed,
    collate_batch,
    consistency_loss,
    is_rank0,
    move_optimizer_state_to_device,
    print_rank0,
    reduce_mean,
    reduce_sum,
    save_checkpoint,
    seed_everything,
    setup_distributed,
)


def load_compatible_state(model: VGAlignSegV3, checkpoint_path: Path, device: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    target = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    skipped = sorted(key for key, value in source.items() if key in target and target[key].shape != value.shape)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    missing = [key for key in missing if not key.startswith("backbone.")]
    print(
        f"Loaded {len(compatible)} compatible tensors from {checkpoint_path}; "
        f"skipped_shape={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return checkpoint if isinstance(checkpoint, dict) else {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VG-AlignSeg V3 on the 8-view result dataset.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "v3_result")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=194)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--eval-split", type=str, choices=("train", "val", "test"), default="test")
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.1)
    parser.add_argument("--focal-loss-weight", type=float, default=0.5)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tversky-loss-weight", type=float, default=0.5)
    parser.add_argument("--boundary-loss-weight", type=float, default=0.2)
    parser.add_argument("--boundary-head-loss-weight", type=float, default=0.05)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.0005)
    parser.add_argument("--consistency-start-step", type=int, default=4000)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--refine-hidden-dim", type=int, default=128)
    parser.add_argument("--no-prototypes", action="store_true")
    parser.add_argument(
        "--use-object-class-prior",
        action="store_true",
        help="Restrict predictions to background plus the actor ids listed for each object.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-optimizer", action="store_true")
    parser.add_argument(
        "--sample-policy",
        choices=("uniform", "actor-count", "rare-actor", "actor-count-rare", "metrics-hard", "hybrid-hard"),
        default="uniform",
        help="Non-uniform object sampler for long-tail and hard-object training.",
    )
    parser.add_argument("--sample-power", type=float, default=0.5)
    parser.add_argument("--rare-power", type=float, default=0.5)
    parser.add_argument("--metrics-hardness-path", type=Path, nargs="*", default=None)
    parser.add_argument("--hardness-target-miou", type=float, default=0.75)
    parser.add_argument("--hardness-power", type=float, default=1.0)
    parser.add_argument("--sampler-max-weight", type=float, default=25.0)
    parser.add_argument("--eval-cleanup", action="store_true")
    parser.add_argument("--eval-cleanup-min-area", type=int, default=16)
    parser.add_argument("--target-eval-miou", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--viz-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser.parse_args()


def boundary_target(target: torch.Tensor) -> torch.Tensor:
    flat = target.reshape(target.shape[0] * target.shape[1], 1, target.shape[2], target.shape[3]).float()
    max_label = F.max_pool2d(flat, kernel_size=3, stride=1, padding=1)
    min_label = -F.max_pool2d(-flat, kernel_size=3, stride=1, padding=1)
    return (max_label != min_label).float().reshape(target.shape[0], target.shape[1], 1, target.shape[2], target.shape[3])


class WeightedDistributedSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        if weights.dim() != 1:
            raise ValueError("weights must be a 1D tensor")
        self.weights = weights.double().clamp_min(1.0e-8)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        sampled = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator).tolist()
        return iter(sampled[self.rank : self.total_size : self.num_replicas])

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def load_metric_hardness(paths: list[Path] | None) -> dict[str, float]:
    if not paths:
        return {}
    values: dict[str, list[float]] = defaultdict(list)
    for path in paths:
        if path is None or not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("phase") != "train" or "object_id" not in row or "mean_iou" not in row:
                    continue
                values[str(row["object_id"])].append(float(row["mean_iou"]))
    return {object_id: sum(scores) / len(scores) for object_id, scores in values.items() if scores}


def build_sampling_weights(args: argparse.Namespace, dataset: ResultMultiViewDataset) -> torch.Tensor:
    if args.sample_policy == "uniform":
        return torch.ones(len(dataset), dtype=torch.float64)

    actor_freq = Counter(actor_id for obj in dataset.objects for actor_id in obj.actor_ids)
    max_freq = max(actor_freq.values()) if actor_freq else 1
    metric_hardness = load_metric_hardness(args.metrics_hardness_path)
    weights = []
    for obj in dataset.objects:
        weight = 1.0
        n_actors = max(1, len(obj.actor_ids))
        if args.sample_policy in {"actor-count", "actor-count-rare", "hybrid-hard"}:
            weight *= n_actors**args.sample_power
        if args.sample_policy in {"rare-actor", "actor-count-rare", "hybrid-hard"}:
            rare_scores = [(max_freq / max(1, actor_freq[actor_id])) ** args.rare_power for actor_id in obj.actor_ids]
            if rare_scores:
                # Mean keeps many-part objects stable; max gives rare tail parts a voice.
                weight *= 0.5 * (sum(rare_scores) / len(rare_scores)) + 0.5 * max(rare_scores)
        if args.sample_policy in {"metrics-hard", "hybrid-hard"} and obj.object_id in metric_hardness:
            miou = metric_hardness[obj.object_id]
            miss = max(0.0, args.hardness_target_miou - miou) / max(1.0e-6, args.hardness_target_miou)
            weight *= (1.0 + miss) ** args.hardness_power
        weights.append(weight)

    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    weight_tensor = weight_tensor / weight_tensor.mean().clamp_min(1.0e-8)
    if args.sampler_max_weight > 0:
        weight_tensor = weight_tensor.clamp(max=args.sampler_max_weight)
    return weight_tensor


def evaluate_subset(
    model: VGAlignSegV3,
    loader: DataLoader,
    cache_dir: Path,
    device: str,
    num_classes: int,
    max_batches: int | None,
    cleanup: bool,
    cleanup_min_area: int,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "pixel_errors": 0,
        "pixel_count": 0,
        "mean_iou_sum": 0.0,
        "clean_pixel_errors": 0,
        "clean_mean_iou_sum": 0.0,
        "batches": 0,
    }
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = batch["images"].to(device)
            target = batch["masks"].to(device=device, dtype=torch.long)
            cached = cached_backbone_outputs(model, images, batch["object_id"], cache_dir, device)
            outputs = model.forward_from_backbone_outputs(images, cached, output_size=images.shape[-2:])
            final_logits = outputs["final_logits"]
            if getattr(model, "use_object_class_prior", False):
                final_logits = mask_logits_to_actor_ids(final_logits, batch["actor_ids"])
            metrics = segmentation_metrics(final_logits, target, num_classes)
            totals["pixel_errors"] += metrics["pixel_errors"]
            totals["pixel_count"] += int(target.numel())
            totals["mean_iou_sum"] += metrics["mean_iou"]
            if cleanup:
                clean_metrics = segmentation_metrics_with_cleanup(
                    final_logits,
                    target,
                    num_classes,
                    min_area=cleanup_min_area,
                    fill_holes=True,
                )
                totals["clean_pixel_errors"] += clean_metrics["pixel_errors"]
                totals["clean_mean_iou_sum"] += clean_metrics["mean_iou"]
            totals["batches"] += 1
    if totals["batches"] == 0:
        return {"eval_pixel_accuracy": 0.0, "eval_mean_iou": 0.0, "eval_pixel_errors": 0, "eval_batches": 0}
    out = {
        "eval_pixel_accuracy": 1.0 - totals["pixel_errors"] / max(1, totals["pixel_count"]),
        "eval_mean_iou": totals["mean_iou_sum"] / totals["batches"],
        "eval_pixel_errors": totals["pixel_errors"],
        "eval_batches": totals["batches"],
    }
    if cleanup:
        out.update(
            {
                "eval_clean_pixel_accuracy": 1.0 - totals["clean_pixel_errors"] / max(1, totals["pixel_count"]),
                "eval_clean_mean_iou": totals["clean_mean_iou_sum"] / totals["batches"],
                "eval_clean_pixel_errors": totals["clean_pixel_errors"],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, local_rank, device = setup_distributed(args)
    seed_everything(args.seed + rank)
    if args.batch_size != 1:
        raise ValueError("Use --batch-size 1 for cached object-level training.")

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
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
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
        shuffle_split=args.shuffle_split,
        split_seed=args.split_seed,
    )
    sample_weights = build_sampling_weights(args, train_set)
    if args.sample_policy == "uniform":
        train_sampler = (
            DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
            if distributed
            else None
        )
    elif distributed:
        train_sampler = WeightedDistributedSampler(sample_weights, num_replicas=world_size, rank=rank, seed=args.seed)
    else:
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_set), replacement=True)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None and args.sample_policy == "uniform",
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
        "model_version": "v3",
        "num_classes": train_set.num_classes,
        "classes": train_set.classes,
        "train_samples": len(train_set),
        "eval_split": args.eval_split,
        "eval_samples": len(eval_set),
        "views": args.views,
        "image_size": args.image_size,
        "distributed": distributed,
        "world_size": world_size,
        "shuffle_split": args.shuffle_split,
        "split_seed": args.split_seed,
        "sample_policy": args.sample_policy,
        "sample_weight_min": float(sample_weights.min().item()) if len(sample_weights) else 0.0,
        "sample_weight_max": float(sample_weights.max().item()) if len(sample_weights) else 0.0,
        "sample_weight_mean": float(sample_weights.mean().item()) if len(sample_weights) else 0.0,
    }
    if is_rank0(rank):
        (args.output_dir / "run_config.json").write_text(json.dumps({"args": vars(args), "meta": meta}, indent=2, default=str))
        train_set.write_manifest(args.output_dir / "train_manifest.json")
        eval_set.write_manifest(args.output_dir / f"{args.eval_split}_manifest.json")
    barrier(distributed)

    base_model = VGAlignSegV3(
        num_classes=train_set.num_classes,
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        min_confidence=args.min_confidence,
        refine_hidden_dim=args.refine_hidden_dim,
        use_prototypes=not args.no_prototypes,
    ).to(device)
    base_model.use_object_class_prior = args.use_object_class_prior
    resume_checkpoint: dict = {}
    if args.resume is not None:
        resume_checkpoint = load_compatible_state(base_model, args.resume, device)
        print_rank0(rank, f"Warm-started from {args.resume}", flush=True)

    base_model.backbone.eval()
    params = [p for name, p in base_model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume_optimizer and resume_checkpoint.get("optimizer") is not None:
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
    global_step = int(resume_checkpoint.get("step", 0)) if args.resume_optimizer else 0
    best_eval_miou = -1.0

    print_rank0(
        rank,
        f"Training VG-AlignSeg V3: train={len(train_set)} {args.eval_split}={len(eval_set)} "
        f"classes={train_set.num_classes} cache={args.cache_dir} world_size={world_size}",
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
            final_logits = outputs["final_logits"]
            coarse_logits = outputs["coarse_logits"]
            if args.use_object_class_prior:
                final_logits = mask_logits_to_actor_ids(final_logits, batch["actor_ids"])
                coarse_logits = mask_logits_to_actor_ids(coarse_logits, batch["actor_ids"])
            final_flat = flatten_logits(final_logits)
            coarse_flat = flatten_logits(coarse_logits)
            target_flat = flatten_target(target)
            weights = class_weights(target.cpu(), train_set.num_classes).to(device)

            final_ce = F.cross_entropy(final_flat, target_flat, weight=weights)
            final_focal = focal_loss(final_flat, target_flat, weight=weights, gamma=args.focal_gamma)
            final_dice = dice_loss(final_flat, target_flat)
            final_tversky = tversky_loss(final_flat, target_flat)
            final_boundary = boundary_loss(final_flat, target_flat)
            coarse_ce = F.cross_entropy(coarse_flat, target_flat, weight=weights)
            boundary_aux = F.binary_cross_entropy_with_logits(outputs["boundary_logits"], boundary_target(target))
            cons = consistency_loss(outputs)
            cons_weight = args.consistency_loss_weight if global_step >= args.consistency_start_step else 0.0
            loss = (
                final_ce
                + args.focal_loss_weight * final_focal
                + args.dice_loss_weight * final_dice
                + args.tversky_loss_weight * final_tversky
                + args.boundary_loss_weight * final_boundary
                + args.boundary_head_loss_weight * boundary_aux
                + args.coarse_loss_weight * coarse_ce
                + cons_weight * cons
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            if global_step == 1 or global_step % args.log_every == 0:
                with torch.no_grad():
                    metrics = segmentation_metrics(final_logits.detach(), target, train_set.num_classes)
                pixel_errors = reduce_sum(metrics["pixel_errors"], device, distributed)
                pixel_count = reduce_sum(target.numel(), device, distributed)
                row = {
                    "step": global_step,
                    "phase": "train",
                    "epoch": epoch,
                    "object_id": batch["object_id"][0],
                    "loss": reduce_mean(loss.item(), device, distributed),
                    "final_ce": reduce_mean(final_ce.item(), device, distributed),
                    "focal": reduce_mean(final_focal.item(), device, distributed),
                    "dice": reduce_mean(final_dice.item(), device, distributed),
                    "tversky": reduce_mean(final_tversky.item(), device, distributed),
                    "boundary": reduce_mean(final_boundary.item(), device, distributed),
                    "boundary_aux": reduce_mean(boundary_aux.item(), device, distributed),
                    "coarse_ce": reduce_mean(coarse_ce.item(), device, distributed),
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
                        f"loss={row['loss']:.4f} ce={row['final_ce']:.4f} focal={row['focal']:.4f} "
                        f"miou={row['mean_iou']:.4f} acc={row['pixel_accuracy']:.4f}",
                        flush=True,
                    )

            if is_rank0(rank) and (global_step == 1 or global_step % args.viz_every == 0):
                save_visualization(
                    args.output_dir / f"train_step_{global_step:06d}.png",
                    images.detach(),
                    target.detach(),
                    final_logits.detach(),
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
                        cleanup=args.eval_cleanup,
                        cleanup_min_area=args.eval_cleanup_min_area,
                    )
                    with metrics_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": global_step, "phase": args.eval_split, "epoch": epoch, **eval_metrics}) + "\n")
                    save_checkpoint(args.output_dir / "latest.pt", base_model, optimizer, global_step, args, meta)
                    score = eval_metrics.get("eval_clean_mean_iou", eval_metrics["eval_mean_iou"])
                    if score > best_eval_miou:
                        best_eval_miou = score
                        save_checkpoint(args.output_dir / "best.pt", base_model, optimizer, global_step, args, meta)
                    clean_msg = ""
                    if "eval_clean_mean_iou" in eval_metrics:
                        clean_msg = f" clean_miou={eval_metrics['eval_clean_mean_iou']:.4f}"
                    print(
                        f"{args.eval_split}@{global_step}: miou={eval_metrics['eval_mean_iou']:.4f}{clean_msg} "
                        f"acc={eval_metrics['eval_pixel_accuracy']:.4f}",
                        flush=True,
                    )
                    if args.target_eval_miou is not None and score >= args.target_eval_miou:
                        print(
                            f"Reached target_eval_miou={args.target_eval_miou:.4f} with score={score:.4f}; stopping.",
                            flush=True,
                        )
                        stop_for_target = True
                    else:
                        stop_for_target = False
                else:
                    stop_for_target = False
                if distributed:
                    flag = torch.tensor(1 if stop_for_target else 0, device=device)
                    dist.broadcast(flag, src=0)
                    stop_for_target = bool(flag.item())
                barrier(distributed)
                if stop_for_target:
                    cleanup_distributed(distributed)
                    return
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
