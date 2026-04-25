import argparse
import os
import sys

import torch


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from models import VGAlignSegV1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VG-AlignSeg V1 sanity check")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = VGAlignSegV1(
        num_classes=args.num_classes,
        checkpoint_path=args.checkpoint_path,
    ).to(device)
    model.eval()

    images = torch.rand(
        args.batch_size,
        args.views,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )

    with torch.no_grad():
        outputs = model(images)

    keys_to_print = [
        "token_grid",
        "point_grid",
        "point_conf_grid",
        "coarse_logits_lowres",
        "coarse_logits",
        "propagated_logits_lowres",
        "propagated_logits",
        "final_logits_lowres",
        "final_logits",
    ]

    print("VG-AlignSeg V1 sanity check passed.")
    for key in keys_to_print:
        print(f"{key}: {tuple(outputs[key].shape)}")


if __name__ == "__main__":
    main()
