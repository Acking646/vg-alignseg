#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="final_results/runs/v4_random2000_top4_independent_seed42"
EVAL_DIR="final_results/evaluations/v4_random2000_top4_dynamic_target_only_top4"

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=2 scripts/train_v4_part_transfer.py \
  --output-dir "$RUN_DIR" \
  --cache-dir data/vggt_cache_result_224 \
  --views 8 \
  --image-size 224 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1 \
  --shuffle-split \
  --split-seed 42 \
  --eval-split test \
  --epochs 0 \
  --max-steps 4000 \
  --batch-size 1 \
  --num-workers 2 \
  --lr 8e-5 \
  --weight-decay 1e-4 \
  --hidden-dim 128 \
  --refinement-iters 2 \
  --train-source-policy largest \
  --train-source-topk 4 \
  --train-logit-merge max \
  --train-independent-sources \
  --train-exclude-source-views-from-loss \
  --bce-loss-weight 1.0 \
  --focal-loss-weight 20.0 \
  --dice-loss-weight 1.0 \
  --tversky-loss-weight 0.5 \
  --boundary-loss-weight 0.2 \
  --boundary-head-loss-weight 0.05 \
  --focal-alpha 0.75 \
  --focal-gamma 2.0 \
  --eval-thresholds=-4,-3,-2,-1.5,-1,-0.5,0 \
  --eval-source-topk 4 \
  --eval-logit-merge max \
  --eval-max-objects 30 \
  --log-every 10 \
  --eval-log-every 10 \
  --viz-every 500 \
  --save-every 1000 \
  --eval-viz-samples 12

python scripts/plot_training_curves.py \
  --metrics "$RUN_DIR/metrics.jsonl" \
  --output-dir "$RUN_DIR/curves"

CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_v4_dynamic_target_only.py \
  --checkpoint "$RUN_DIR/best.pt" \
  --output-dir "$EVAL_DIR" \
  --cache-dir data/vggt_cache_result_224 \
  --views 8 \
  --image-size 224 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1 \
  --split test \
  --shuffle-split \
  --split-seed 42 \
  --source-topk 4 \
  --thresholds=-4,-3,-2,-1.5,-1,-0.5,0 \
  --logit-merge max \
  --viz-samples 80 \
  --log-every 10 \
  --device cuda

