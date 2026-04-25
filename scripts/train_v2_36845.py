import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.views_36845 import load_36845_sample  # noqa: E402
from models import VGAlignSegV2  # noqa: E402
from scripts.eval_utils import (  # noqa: E402
    class_weights,
    dice_loss,
    flatten_logits,
    flatten_target,
    save_visualization,
    segmentation_metrics,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "36845_views"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VG-AlignSeg V2 on the local 36845 multi-view sample.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "v2_36845")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--mask-mode", choices=["actors", "chair-semantic"], default="actors")
    parser.add_argument("--actor-ids", type=int, nargs="*", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.0)
    parser.add_argument("--dice-loss-weight", type=float, default=0.5)
    parser.add_argument("--consistency-loss-weight", type=float, default=0.03)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--refine-hidden-dim", type=int, default=128)
    parser.add_argument("--no-prototypes", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--viz-every", type=int, default=250)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_state_dict(model: VGAlignSegV2) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if not k.startswith("backbone.")}


def load_trainable_state(model: VGAlignSegV2, checkpoint_path: Path, device: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    missing = [k for k in missing if not k.startswith("backbone.")]
    if unexpected or missing:
        print(f"Resume state loaded with missing={missing}, unexpected={unexpected}", flush=True)


def consistency_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    final = outputs["final_logits"]
    propagated = outputs["propagated_logits_lowres"]
    bsz, n_views, num_classes, height, width = final.shape
    grid_h, grid_w = propagated.shape[2:4]

    flat_final = final.reshape(bsz * n_views, num_classes, height, width)
    final_low = F.adaptive_avg_pool2d(flat_final, (grid_h, grid_w))
    final_low = final_low.reshape(bsz, n_views, num_classes, grid_h, grid_w).permute(0, 1, 3, 4, 2)

    log_p = F.log_softmax(final_low, dim=-1)
    q = F.softmax(propagated.detach(), dim=-1)
    return F.kl_div(log_p, q, reduction="batchmean")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    images, target, meta = load_36845_sample(
        args.data_root,
        views=args.views,
        image_size=args.image_size,
        mask_mode=args.mask_mode,
        actor_ids=args.actor_ids,
    )
    num_classes = len(meta["classes"])
    images = images.to(device)
    target = target.to(device=device, dtype=torch.long)
    weights = class_weights(target.cpu(), num_classes).to(device)

    model = VGAlignSegV2(
        num_classes=num_classes,
        checkpoint_path=args.checkpoint_path,
        topk=args.topk,
        min_confidence=args.min_confidence,
        refine_hidden_dim=args.refine_hidden_dim,
        use_prototypes=not args.no_prototypes,
    ).to(device)

    if args.resume is not None:
        load_trainable_state(model, args.resume, device)
        print(f"Resumed V2 heads from {args.resume}", flush=True)

    print(
        f"Loaded sample images={tuple(images.shape)} target={tuple(target.shape)} "
        f"classes={num_classes} mode={args.mask_mode}",
        flush=True,
    )
    print("Classes:", json.dumps(meta["classes"], ensure_ascii=False), flush=True)

    print("Caching frozen VGGT outputs...", flush=True)
    model.eval()
    with torch.no_grad():
        cached = model.backbone(images)
        cached = {k: v.detach() for k, v in cached.items() if k in {"token_grid", "point_grid", "point_conf_grid"}}
    model.backbone.to("cpu")
    if device == "cuda":
        torch.cuda.empty_cache()

    for name, module in model.named_children():
        if name != "backbone":
            module.train()
    params = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    (args.output_dir / "run_config.json").write_text(
        json.dumps({"args": vars(args), "meta": meta}, indent=2, default=str),
        encoding="utf-8",
    )
    metrics_log = args.output_dir / "metrics.jsonl"

    best_errors = None
    best_state = None
    best_step = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_from_backbone_outputs(images, cached, output_size=(args.image_size, args.image_size))

        final_flat = flatten_logits(outputs["final_logits"])
        coarse_flat = flatten_logits(outputs["coarse_logits"])
        target_flat = flatten_target(target)

        final_ce = F.cross_entropy(final_flat, target_flat, weight=weights)
        coarse_ce = F.cross_entropy(coarse_flat, target_flat, weight=weights)
        final_dice = dice_loss(final_flat, target_flat)
        cons = consistency_loss(outputs)
        loss = (
            final_ce
            + args.coarse_loss_weight * coarse_ce
            + args.dice_loss_weight * final_dice
            + args.consistency_loss_weight * cons
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = segmentation_metrics(outputs["final_logits"].detach(), target, num_classes)
            if best_errors is None or metrics["pixel_errors"] < best_errors:
                best_errors = metrics["pixel_errors"]
                best_step = step
                best_state = trainable_state_dict(model)
            row = {
                "step": step,
                "loss": float(loss.item()),
                "final_ce": float(final_ce.item()),
                "coarse_ce": float(coarse_ce.item()),
                "dice": float(final_dice.item()),
                "consistency": float(cons.item()),
                "best_errors": best_errors,
                "best_step": best_step,
                **metrics,
            }
            with metrics_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"step={step:04d} loss={loss.item():.5f} ce={final_ce.item():.5f} "
                f"dice={final_dice.item():.5f} cons={cons.item():.5f} "
                f"miou={metrics['mean_iou']:.5f} acc={metrics['pixel_accuracy']:.5f} "
                f"errors={metrics['pixel_errors']} best_errors={best_errors} best_step={best_step}",
                flush=True,
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
            "model": trainable_state_dict(model),
            "args": vars(args),
            "meta": meta,
        },
        args.output_dir / "v2_36845_checkpoint.pt",
    )
    if best_state is not None:
        torch.save(
            {
                "model": best_state,
                "args": vars(args),
                "meta": meta,
                "best_errors": best_errors,
                "best_step": best_step,
            },
            args.output_dir / "v2_36845_best.pt",
        )
    print(f"Saved V2 checkpoint and outputs under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
