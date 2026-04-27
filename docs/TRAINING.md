# Training VG-AlignSeg V4

This document records the exact training entry point for the V4 source-guided
part transfer model used by the final result package.

## Preconditions

Expected local files:

- Dataset: `data/vg-alignseg-dataset`
- VGGT cache: `data/vggt_cache_result_224`
- Optional resume checkpoint:
  `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`

Checkpoint files and dataset/cache directories are intentionally ignored by git.
The dataset source and final 2000/187 split are documented in
`docs/DATASET_SPLIT.md`.

## Recommended Launch

Use tmux so the run survives terminal disconnects:

```bash
tmux new -s vgtrain

conda activate mova
cd /home/lyx/curriculum/computer_vision/VG-AlignSeg

CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_v4_final.sh outputs/v4_result_ddp2_new_train
```

Detach from tmux with `Ctrl-b d`; reattach with:

```bash
tmux attach -t vgtrain
```

## Fine-Tune From The Strong Checkpoint

To continue from the best historical V4 checkpoint:

```bash
RESUME_CHECKPOINT=outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt \
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_v4_final.sh outputs/v4_result_ddp2_new_finetune
```

By default the launcher uses two GPUs, `8000` steps, top2 source prompts during
training, and top4 source prompts during evaluation. You can override a few
common knobs with environment variables:

```bash
MAX_STEPS=12000 LR=2e-5 SAVE_EVERY=500 VIZ_EVERY=250 \
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_v4_final.sh outputs/v4_result_ddp2_long
```

## What The Launcher Runs

`scripts/train_v4_final.sh` wraps this core training command:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_v4_part_transfer.py \
  --output-dir outputs/v4_result_ddp2_new_train \
  --data-root data/vg-alignseg-dataset \
  --cache-dir data/vggt_cache_result_224 \
  --views 8 \
  --image-size 224 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1 \
  --eval-split test \
  --epochs 0 \
  --max-steps 8000 \
  --batch-size 1 \
  --num-workers 2 \
  --lr 2e-5 \
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
  --eval-viz-samples 40 \
  --log-every 10 \
  --eval-log-every 25 \
  --viz-every 250 \
  --save-every 500
```

The run writes:

- `latest.pt`
- `best.pt`
- `metrics.jsonl`
- periodic train visualizations
- periodic test visualizations
- `train_manifest.json`
- `test_manifest.json`
- `run_config.json`

## Plot Curves

After training:

```bash
python scripts/plot_training_curves.py \
  --metrics outputs/v4_result_ddp2_new_train/metrics.jsonl \
  --output-dir outputs/v4_result_ddp2_new_train/curves
```

## Evaluate Strict Target-Only Top4

Use the same final evaluation protocol reported in `final_results`:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_v4_dynamic_target_only.py \
  --checkpoint outputs/v4_result_ddp2_new_train/best.pt \
  --output-dir outputs/v4_result_ddp2_new_train_eval_target_only_top4 \
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
  --eval-dir outputs/v4_result_ddp2_new_train_eval_target_only_top4 \
  --category-map final_results/metadata/object_category_map.json
```

The main final metrics live in:

- `final_results/evaluations/v4_target_only_top4_best4500/summary.json`
- `final_results/final_metrics_table.md`
