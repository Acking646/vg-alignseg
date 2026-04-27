#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt"
TARGET_DIR="final_results/evaluations/v4_target_only_top4_best4500"
SOURCE_COPY_DIR="final_results/evaluations/v4_staged_oracle_top4_best4500_fixedviz"
TRAINING_DIR="final_results/training/v4_top2_phase2_random_copy"

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_v4_dynamic_target_only.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$TARGET_DIR" \
  --cache-dir data/vggt_cache_result_224 \
  --views 8 \
  --image-size 224 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1 \
  --split test \
  --source-topk 4 \
  --thresholds=-4,-3.5,-3,-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,0 \
  --logit-merge max \
  --viz-samples 80 \
  --log-every 25 \
  --device cuda
python scripts/summarize_target_only_results.py \
  --eval-dir "$TARGET_DIR" \
  --category-map final_results/metadata/object_category_map.json

python scripts/plot_training_curves.py \
  --metrics "$TRAINING_DIR/metrics.jsonl" \
  --output-dir "$TRAINING_DIR/curves"

# Optional source-copy reference. This reproduces the old fixedviz-style upper
# guided-composition visualization; it is not the main strict target-only metric.
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_v4_staged_oracle.py \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$SOURCE_COPY_DIR" \
  --cache-dir data/vggt_cache_result_224 \
  --category-list final_results/metadata/category_models_list.txt \
  --views 8 \
  --image-size 224 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1 \
  --split test \
  --source-topk 4 \
  --thresholds=-4,-3.5,-3,-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,0 \
  --logit-merge max \
  --copy-source-views \
  --viz-samples 80 \
  --log-every 25 \
  --device cuda
