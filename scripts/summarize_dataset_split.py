from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.result_multiview import VIEW_ORDER, list_result_objects, split_objects  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the VG-AlignSeg dataset source and train/test split.")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "vg-alignseg-dataset")
    parser.add_argument("--category-map", type=Path, default=REPO_ROOT / "final_results" / "metadata" / "object_category_map.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "final_results" / "split_summary.json")
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=-1)
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def optional_count(value: int | None) -> int | None:
    if value is None or value < 0:
        return None
    return value


def split_stats(rows, category_map: dict[str, str]) -> dict:
    if not rows:
        return {"num_objects": 0}
    category_counts = Counter(category_map.get(obj.object_id, "Unknown") for obj in rows)
    actor_counts = [len(obj.actor_ids) for obj in rows]
    return {
        "num_objects": len(rows),
        "first_10_object_ids": [obj.object_id for obj in rows[:10]],
        "last_10_object_ids": [obj.object_id for obj in rows[-10:]],
        "num_categories": len(category_counts),
        "category_counts": dict(sorted(category_counts.items())),
        "actor_count_min": min(actor_counts),
        "actor_count_max": max(actor_counts),
        "actor_count_mean": sum(actor_counts) / len(actor_counts),
    }


def main() -> None:
    args = parse_args()
    category_map = json.loads(args.category_map.read_text(encoding="utf-8")) if args.category_map.exists() else {}
    objects = list_result_objects(args.data_root, views=args.views)
    split_kwargs = {
        "train_count": optional_count(args.train_count),
        "val_count": optional_count(args.val_count),
        "test_count": optional_count(args.test_count),
        "shuffle": args.shuffle_split,
        "seed": args.split_seed,
    }
    train = split_objects(objects, "train", **split_kwargs)
    val = split_objects(objects, "val", **split_kwargs)
    test = split_objects(objects, "test", **split_kwargs)

    summary = {
        "dataset_source": "Hugging Face dataset luyu1021/vg-alignseg, file result.zip",
        "dataset_url": "https://huggingface.co/datasets/luyu1021/vg-alignseg",
        "local_root": str(args.data_root),
        "loader": "data.result_multiview.ResultMultiViewDataset",
        "valid_object_rule": (
            "Directory name ends with _views, has color/ and part_mask/, contains the first "
            f"{args.views} expected color view files, and has at least one part_mask/actor_* directory."
        ),
        "view_order": VIEW_ORDER[: args.views],
        "split_policy": "shuffled by seed" if args.shuffle_split else "lexicographic directory order, no shuffle",
        "split_args": {
            "train_count": args.train_count,
            "val_count": args.val_count,
            "test_count": args.test_count,
            "shuffle_split": args.shuffle_split,
            "split_seed": args.split_seed,
        },
        "total_valid_objects": len(objects),
        "train": split_stats(train, category_map),
        "val": split_stats(val, category_map),
        "test": split_stats(test, category_map),
        "manifest_files": {
            "train": "final_results/training/v4_top2_phase2_random_copy/train_manifest.json",
            "test": "final_results/evaluations/v4_target_only_top4_best4500/test_manifest.json",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
