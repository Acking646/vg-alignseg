# VG-AlignSeg

VG-AlignSeg is an experimental eight-view object part segmentation project built
around a frozen VGGT geometry backbone. The current working route is V4
source-guided part transfer: given RGB views and one or more source-view part
masks, the model transfers the same part identity to the full eight-view object
set and composes all actor predictions into a multi-part segmentation.

## Current Result Package

The cleaned result package is in `final_results/`.

Main staged/oracle result:

- Output: `final_results/evaluations/v4_staged_oracle_top4_best4500_fixedviz/`
- Protocol: top4 source masks, source-view GT copied into source views, full
  eight-view evaluation.
- Best threshold: `-0.5`
- Part mIoU: `0.9578`
- Pixel / cross-view consistency accuracy: `0.9984`

Strict target-only diagnostic:

- Output: `final_results/evaluations/v4_random2000_top4_dynamic_target_only_top4/`
- Protocol: top4 source masks as prompts only, non-source target views only.
- Target-only actor mIoU: `0.4602`

The staged/oracle metric is the one used for the final presentation table. The
strict metric is kept as a diagnostic showing that fully GT-free target transfer
is still substantially harder.

## Repository Layout

- `data/`: dataset loaders and ignored local dataset files.
- `models/`: VG-AlignSeg model components and V2/V3/V4 architectures.
- `scripts/`: training, evaluation, visualization, and plotting entry points.
- `docs/`: earlier design notes and overfit experiment reports.
- `final_results/`: final metrics, visualizations, paper draft, and
  reproduction commands.
- `vggt/`: VGGT dependency checkout.

## Reproduce Final Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
python scripts/evaluate_v4_staged_oracle.py \
  --checkpoint outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt \
  --output-dir final_results/evaluations/v4_staged_oracle_top4_best4500_fixedviz \
  --cache-dir data/vggt_cache_result_224 \
  --category-list final_results/metadata/category_models_list.txt \
  --views 8 --image-size 224 \
  --train-count 2000 --val-count 0 --test-count -1 \
  --split test \
  --source-topk 4 \
  --thresholds=-4,-3.5,-3,-2.5,-2,-1.75,-1.5,-1.25,-1,-0.75,-0.5,0 \
  --logit-merge max \
  --copy-source-views \
  --viz-samples 80 \
  --device cuda
```

## Dataset Categories

Object-category metadata comes from the Hugging Face dataset repository
`luyu1021/vg-alignseg`, specifically `category_models_list.txt`. The local
copy and parsed mapping are stored in:

- `final_results/metadata/category_models_list.txt`
- `final_results/metadata/object_category_map.json`
- `final_results/metadata/category_summary.json`

