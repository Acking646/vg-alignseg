from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add category, granularity, and four-table metrics to a strict target-only V4 eval directory."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--category-map", type=Path, required=True)
    return parser.parse_args()


def actor_count_bucket(actor_count: int) -> str:
    if actor_count <= 2:
        return "coarse_1_2_parts"
    if actor_count <= 4:
        return "medium_3_4_parts"
    return "fine_5plus_parts"


def mean(values: list[float]) -> float:
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


def best_threshold_key(summary: dict) -> str:
    threshold = summary["best_threshold"]
    key = str(threshold)
    if key in summary.get("threshold_metrics", {}):
        return key
    return f"{float(threshold):.1f}"


def enrich_rows(per_object: list[dict], category_map: dict[str, str], threshold_key: str) -> list[dict]:
    enriched = []
    for row in per_object:
        object_id = str(row["object_id"])
        score = row["scores"][threshold_key]
        sources = row.get("sources_by_class", {})
        actor_count = int(score.get("actors", len(sources)))
        enriched.append(
            {
                "index": int(row["index"]),
                "object_id": object_id,
                "category": category_map.get(object_id, "Unknown"),
                "actor_count": actor_count,
                "granularity": actor_count_bucket(actor_count),
                "mean_iou": float(score["mean_iou"]),
                "pixel_accuracy": float(score["pixel_accuracy"]),
                "pixel_errors": int(score["pixel_errors"]),
            }
        )
    return enriched


def main() -> None:
    args = parse_args()
    summary_path = args.eval_dir / "summary.json"
    per_object_path = args.eval_dir / "per_object.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not per_object_path.exists():
        raise FileNotFoundError(per_object_path)
    if not args.category_map.exists():
        raise FileNotFoundError(args.category_map)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_object = json.loads(per_object_path.read_text(encoding="utf-8"))
    category_map = json.loads(args.category_map.read_text(encoding="utf-8"))
    threshold_key = best_threshold_key(summary)
    enriched = enrich_rows(per_object, category_map, threshold_key)
    per_category = summarize_groups(enriched, "category")
    per_granularity = summarize_groups(enriched, "granularity")

    table_metrics = {
        "iou_object_category": mean([row["mean_iou"] for row in per_category.values()]),
        "iou_granularity": mean([row["mean_iou"] for row in per_granularity.values()]),
        "iou_part": float(summary["eval_mean_iou"]),
        "cross_view_consistency_acc": float(summary["eval_pixel_accuracy"]),
    }
    summary["table_metrics"] = table_metrics
    summary["category_map"] = str(args.category_map)
    summary["metric_note"] = "Strict target-only top-k source-guided transfer; source GT is not copied into metrics."

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.eval_dir / "per_object_enriched.json").write_text(json.dumps(enriched, indent=2), encoding="utf-8")
    (args.eval_dir / "per_category.json").write_text(json.dumps(per_category, indent=2), encoding="utf-8")
    (args.eval_dir / "per_granularity.json").write_text(json.dumps(per_granularity, indent=2), encoding="utf-8")
    print(json.dumps({"table_metrics": table_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
