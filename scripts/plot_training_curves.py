import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VG-AlignSeg training curves from metrics.jsonl.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smooth", type=int, default=25)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def values(rows: Iterable[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        steps.append(int(row["step"]))
        vals.append(float(value))
    return steps, vals


def smoothed(vals: list[float], window: int) -> list[float]:
    if window <= 1 or len(vals) < window:
        return vals
    out: list[float] = []
    acc = 0.0
    for idx, value in enumerate(vals):
        acc += value
        if idx >= window:
            acc -= vals[idx - window]
            out.append(acc / window)
        else:
            out.append(acc / (idx + 1))
    return out


def plot_series(ax, rows: list[dict[str, Any]], key: str, label: str, smooth: int = 1) -> None:
    steps, vals = values(rows, key)
    if not vals:
        return
    ax.plot(steps, vals, alpha=0.22, linewidth=0.9)
    ax.plot(steps, smoothed(vals, smooth), label=label, linewidth=1.7)


def write_summary(path: Path, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    summary: dict[str, Any] = {
        "train_points": len(train_rows),
        "eval_points": len(eval_rows),
        "last_train": train_rows[-1] if train_rows else None,
        "last_eval": eval_rows[-1] if eval_rows else None,
    }
    if eval_rows:
        summary["best_eval_mean_iou"] = max(float(r.get("eval_mean_iou", r.get("val_mean_iou", 0.0))) for r in eval_rows)
        summary["best_eval_pixel_accuracy"] = max(
            float(r.get("eval_pixel_accuracy", r.get("val_pixel_accuracy", 0.0))) for r in eval_rows
        )
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.metrics)
    output_dir = args.output_dir or args.metrics.parent / "curves"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        write_summary(output_dir / "summary.json", [], [])
        print(f"No metrics yet: {args.metrics}")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_rows = [row for row in rows if row.get("phase") == "train" or "loss" in row]
    eval_rows = [
        row
        for row in rows
        if row.get("phase") in {"val", "test"} or "eval_mean_iou" in row or "val_mean_iou" in row
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for key, label in [
        ("loss", "total loss"),
        ("final_ce", "final CE"),
        ("coarse_ce", "coarse CE"),
        ("dice", "dice"),
        ("consistency", "consistency"),
    ]:
        plot_series(ax, train_rows, key, label, smooth=args.smooth)
    ax.set_title("Training Loss")
    ax.set_xlabel("step")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, train_rows, "mean_iou", "train mIoU", smooth=args.smooth)
    plot_series(ax, train_rows, "pixel_accuracy", "train pixel acc", smooth=args.smooth)
    plot_series(ax, eval_rows, "eval_mean_iou", "test mIoU", smooth=1)
    plot_series(ax, eval_rows, "eval_pixel_accuracy", "test pixel acc", smooth=1)
    plot_series(ax, eval_rows, "eval_clean_mean_iou", "test clean mIoU", smooth=1)
    plot_series(ax, eval_rows, "eval_clean_pixel_accuracy", "test clean pixel acc", smooth=1)
    plot_series(ax, eval_rows, "val_mean_iou", "val mIoU", smooth=1)
    plot_series(ax, eval_rows, "val_pixel_accuracy", "val pixel acc", smooth=1)
    ax.set_title("Segmentation Quality")
    ax.set_xlabel("step")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "miou_accuracy.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_series(ax, train_rows, "pixel_errors", "train pixel errors", smooth=args.smooth)
    plot_series(ax, eval_rows, "eval_pixel_errors", "test pixel errors", smooth=1)
    plot_series(ax, eval_rows, "val_pixel_errors", "val pixel errors", smooth=1)
    ax.set_title("Pixel Errors")
    ax.set_xlabel("step")
    ax.set_yscale("symlog")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pixel_errors.png", dpi=160)
    plt.close(fig)

    write_summary(output_dir / "summary.json", train_rows, eval_rows)
    print(f"Wrote curves to {output_dir}")


if __name__ == "__main__":
    main()
