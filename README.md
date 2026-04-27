# VG-AlignSeg

VG-AlignSeg is an eight-view object part segmentation project built around a
frozen VGGT geometry backbone. The final cleaned version uses a V4
source-guided part transfer formulation: given RGB views and top-k source-view
part masks, the model predicts the same local actor part on the remaining
views, then composes all actor predictions into a multi-part segmentation.

## Final Result

The main result package is in `final_results/`.

Primary strict target-only evaluation:

- Output: `final_results/evaluations/v4_target_only_top4_best4500/`
- Visualizations:
  `final_results/evaluations/v4_target_only_top4_best4500/visualizations/`
- Checkpoint:
  `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`
- Protocol: for every actor, select the top-4 visible source views by GT mask
  area, use those masks as prompts, and evaluate only the non-source target
  views. Source GT is not copied into the metric.
- Best threshold: `-0.5`

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 56.14 | 53.44 | 61.71 | 99.75 |

Auxiliary source-copy reference:

- Output:
  `final_results/evaluations/v4_staged_oracle_top4_best4500_fixedviz/`
- Protocol: same top4 source masks, but source-view GT is copied into source
  views and the metric is computed over all eight views.
- Result: 92.42 category IoU, 95.88 granularity IoU, 95.78 part mIoU, 99.84
  pixel accuracy.
- This is kept as an upper/reference visualization setting, not the main strict
  target-only metric.

## Repository Layout

- `data/`: dataset loaders and ignored local dataset/cache files.
- `models/`: VG-AlignSeg model components and V2/V3/V4 architectures.
- `scripts/`: training, evaluation, visualization, plotting, and result summary
  entry points.
- `docs/`: earlier design notes and overfit experiment reports.
- `final_results/`: final metrics, visualizations, paper draft, and
  reproduction commands.
- `vggt/`: VGGT dependency checkout and local weights directory.

## Train V4

The shortest training entry point is:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./scripts/train_v4_final.sh outputs/v4_result_ddp2_new_train
```

For a tmux workflow, resume/fine-tune options, curve plotting, and final
target-only evaluation commands, see `docs/TRAINING.md`.

## Reproduce Main Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_v4_dynamic_target_only.py \
  --checkpoint outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt \
  --output-dir final_results/evaluations/v4_target_only_top4_best4500 \
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
  --eval-dir final_results/evaluations/v4_target_only_top4_best4500 \
  --category-map final_results/metadata/object_category_map.json
```

## Dataset Categories

Object-category metadata comes from the Hugging Face dataset repository
`luyu1021/vg-alignseg`, specifically `category_models_list.txt`. The local copy
and parsed mapping are stored in:

- `final_results/metadata/category_models_list.txt`
- `final_results/metadata/object_category_map.json`
- `final_results/metadata/category_summary.json`
