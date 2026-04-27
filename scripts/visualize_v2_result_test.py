import argparse
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import ResultMultiViewDataset  # noqa: E402
from models import VGAlignSegV2, VGAlignSegV3  # noqa: E402
from scripts.eval_utils import mask_logits_to_actor_ids, save_visualization, segmentation_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize VG-AlignSeg V2 predictions on result test objects.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "vggt_cache_result_224")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, choices=("val", "test"), default="test")
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=11)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--use-object-class-prior", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_cached(cache_dir: Path, object_id: str, device: str) -> dict[str, torch.Tensor]:
    cache_path = cache_dir / f"{object_id}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing cached VGGT outputs for object {object_id}: {cache_path}")
    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
    return {key: value.to(device=device, dtype=torch.float32) for key, value in cached.items()}


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    train_args = checkpoint["args"]
    meta = checkpoint["meta"]
    output_dir = args.output_dir or args.checkpoint.parent / f"{args.split}_visualizations_best"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ResultMultiViewDataset(
        args.data_root,
        split=args.split,
        views=int(train_args["views"]),
        image_size=int(train_args["image_size"]),
        train_count=int(train_args.get("train_count", 2000)),
        val_count=int(train_args.get("val_count", 0)),
        test_count=int(train_args.get("test_count", 194)),
        shuffle_split=args.shuffle_split or bool(train_args.get("shuffle_split", False)),
        split_seed=int(train_args.get("split_seed", args.split_seed)),
    )
    num_classes = int(meta["num_classes"])

    model_cls = VGAlignSegV3 if meta.get("model_version") == "v3" or train_args.get("model_version") == "v3" else VGAlignSegV2
    model = model_cls(
        num_classes=num_classes,
        checkpoint_path=train_args.get("checkpoint_path"),
        topk=int(train_args["topk"]),
        min_confidence=float(train_args["min_confidence"]),
        refine_hidden_dim=int(train_args["refine_hidden_dim"]),
        use_prototypes=not bool(train_args.get("no_prototypes", False)),
    ).to(args.device)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()

    rows = []
    end = min(len(dataset), args.start_index + args.num_samples)
    with torch.no_grad():
        for idx in range(args.start_index, end):
            item = dataset[idx]
            object_id = item["object_id"]
            images = item["images"].unsqueeze(0).to(args.device)
            target = item["masks"].unsqueeze(0).to(device=args.device, dtype=torch.long)
            cached = load_cached(args.cache_dir, object_id, args.device)
            outputs = model.forward_from_backbone_outputs(images, cached, output_size=images.shape[-2:])
            final_logits = outputs["final_logits"]
            if args.use_object_class_prior or bool(train_args.get("use_object_class_prior", False)):
                final_logits = mask_logits_to_actor_ids(final_logits, [item["actor_ids"]])
            metrics = segmentation_metrics(final_logits, target, num_classes)

            out_path = output_dir / f"{idx:04d}_{object_id}.png"
            save_visualization(out_path, images.cpu(), target.cpu(), final_logits.cpu(), item["view_names"])
            rows.append(
                {
                    "index": idx,
                    "object_id": object_id,
                    "image": out_path.name,
                    "pixel_accuracy": metrics["pixel_accuracy"],
                    "mean_iou": metrics["mean_iou"],
                    "pixel_errors": metrics["pixel_errors"],
                }
            )
            print(f"{idx:04d} {object_id} miou={metrics['mean_iou']:.4f} acc={metrics['pixel_accuracy']:.4f}", flush=True)

    (output_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} visualizations to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
