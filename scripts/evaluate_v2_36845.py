import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.views_36845 import load_36845_sample  # noqa: E402
from models import VGAlignSegV2  # noqa: E402
from scripts.eval_utils import save_visualization, segmentation_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a VG-AlignSeg V2 checkpoint on 36845 views.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "36845_views")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


def lowres_consistency(outputs: dict) -> float:
    final = outputs["final_logits"]
    propagated = outputs["propagated_logits_lowres"]
    bsz, n_views, num_classes, height, width = final.shape
    grid_h, grid_w = propagated.shape[2:4]
    final_low = F.adaptive_avg_pool2d(final.reshape(bsz * n_views, num_classes, height, width), (grid_h, grid_w))
    final_low = final_low.reshape(bsz, n_views, num_classes, grid_h, grid_w).permute(0, 1, 3, 4, 2)
    p = F.softmax(final_low, dim=-1)
    q = F.softmax(propagated, dim=-1)
    agreement = (p.argmax(dim=-1) == q.argmax(dim=-1)).float().mean()
    kl = F.kl_div(F.log_softmax(final_low, dim=-1), q, reduction="batchmean")
    return float(agreement.item()), float(kl.item())


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint["args"]
    meta = checkpoint["meta"]
    output_dir = args.output_dir or args.checkpoint.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    images, target, _ = load_36845_sample(
        args.data_root,
        views=train_args["views"],
        image_size=train_args["image_size"],
        mask_mode=train_args["mask_mode"],
        actor_ids=train_args.get("actor_ids"),
    )
    images = images.to(device)
    target = target.to(device=device, dtype=torch.long)
    num_classes = len(meta["classes"])

    model = VGAlignSegV2(
        num_classes=num_classes,
        checkpoint_path=args.checkpoint_path,
        topk=train_args["topk"],
        min_confidence=train_args["min_confidence"],
        refine_hidden_dim=train_args["refine_hidden_dim"],
        use_prototypes=not train_args.get("no_prototypes", False),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    metrics = segmentation_metrics(outputs["final_logits"], target, num_classes)
    agreement, consistency_kl = lowres_consistency(outputs)
    metrics["cross_view_lowres_agreement"] = agreement
    metrics["cross_view_lowres_kl"] = consistency_kl
    metrics["classes"] = meta["classes"]

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_visualization(output_dir / "prediction.png", images, target, outputs["final_logits"], meta["view_names"])

    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
