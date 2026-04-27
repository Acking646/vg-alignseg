#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-outputs/v4_result_ddp2_new_train}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MAX_STEPS="${MAX_STEPS:-8000}"
LR="${LR:-2e-5}"
SAVE_EVERY="${SAVE_EVERY:-500}"
VIZ_EVERY="${VIZ_EVERY:-250}"
EVAL_VIZ_SAMPLES="${EVAL_VIZ_SAMPLES:-40}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS=(--resume "$RESUME_CHECKPOINT")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
PYTHONUNBUFFERED=1 \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" scripts/train_v4_part_transfer.py \
  --output-dir "$OUTPUT_DIR" \
  --data-root data/vg-alignseg-dataset \
  --cache-dir data/vggt_cache_result_224 \
  --views 8 \
  --image-size 224 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1 \
  --eval-split test \
  --epochs 0 \
  --max-steps "$MAX_STEPS" \
  --batch-size 1 \
  --num-workers 2 \
  --lr "$LR" \
  --weight-decay 1e-4 \
  --hidden-dim 128 \
  --refinement-iters 2 \
  --train-source-policy random \
  --train-source-topk 2 \
  --train-logit-merge max \
  --train-copy-source-views \
  --bce-loss-weight 1.0 \
  --focal-loss-weight 20.0 \
  --dice-loss-weight 1.0 \
  --tversky-loss-weight 0.5 \
  --boundary-loss-weight 0.2 \
  --boundary-head-loss-weight 0.05 \
  --focal-alpha 0.75 \
  --focal-gamma 2.0 \
  --compose-threshold -2.0 \
  --eval-thresholds=-4,-3.5,-3,-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,0 \
  --eval-source-topk 4 \
  --eval-logit-merge max \
  --eval-viz-samples "$EVAL_VIZ_SAMPLES" \
  --log-every 10 \
  --eval-log-every 25 \
  --viz-every "$VIZ_EVERY" \
  --save-every "$SAVE_EVERY" \
  "${RESUME_ARGS[@]}"
