# External Baseline Testing Report

## Dataset And Split

All experiments use the VG-AlignSeg dataset at
`data/vg-alignseg-dataset`. The loader keeps valid `*_views` objects with 8 RGB
views and actor masks, then uses lexicographic order with no shuffle:

- train: first 2000 objects
- validation: 0 objects
- test: remaining 187 objects
- image size: 224 x 224
- views: 8

## Metrics

The reported table uses the same four metrics as the final VG-AlignSeg result:

- `iou-object-category`: category-balanced macro average. For each object
  category, average object-level actor mIoU over all test objects in that
  category; then average those category means.
- `iou-granularity`: granularity-balanced macro average. Objects are bucketed by
  actor count into `coarse_1_2_parts`, `medium_3_4_parts`, and
  `fine_5plus_parts`; average object-level actor mIoU inside each bucket, then
  average the bucket means.
- `iou-part`: mean object-level actor/part IoU. For each object, compute IoU
  for every foreground actor/part and average actors; then average objects.
- `cross-view consistency acc`: pixel accuracy on the evaluated views/pixels.
  For full-view baselines this is multiclass pixel accuracy over all 8 views.
  For strict source-guided VG-AlignSeg V4 this is per-actor binary pixel
  accuracy on non-source target views only.

Detailed formulas are in `docs/METRICS.md`.

## Methods

### VG-AlignSeg V4 Reference

This is the existing strict target-only source-guided result:
`final_results/evaluations/v4_target_only_top4_best4500/summary.json`.
It uses top-4 source prompts and evaluates only predicted target-view pixels
without copying GT masks into evaluated views.

### PanSt3R

PanSt3R is evaluated from the official repository at
`/home/lyx/curriculum/computer_vision/panst3r` with checkpoint
`/home/lyx/curriculum/computer_vision/baseline_weights/panst3r/panst3r_v2_512_5ds.pth`.

Because PanSt3R outputs category-level panoptic segment ids rather than
VG-AlignSeg actor ids, predicted segments are matched to GT actors with
Hungarian matching for scoring only. No GT mask is copied into the prediction.

Command:

```bash
PYTHONUNBUFFERED=1 /home/lyx/miniconda3/envs/panst3r_eval/bin/python \
  scripts/evaluate_panst3r_baseline.py \
  --output-dir final_results/baselines/panst3r/v2_224_oracle_match \
  --device cuda:0 \
  --image-size 224 \
  --num-keyframes 8 \
  --max-bs 1 \
  --amp bf16 \
  --log-every 10 \
  --viz-samples 40
```

Output:
`final_results/baselines/panst3r/v2_224_oracle_match/`

### Cross View Transformers

The original Cross View Transformers repository is at
`/home/lyx/curriculum/computer_vision/cross_view_transformers`. It is a
nuScenes/Argoverse BEV map-view semantic segmentation model. Its native input is
calibrated vehicle cameras plus ego geometry, and its native output is
`pred["bev"]`, not image-space part masks. Therefore the original model is not a
valid direct baseline for VG-AlignSeg.

To make a reasonable task-compatible comparison, I added
`scripts/train_cvt_image_adapter.py`: a lightweight image-space adapter inspired
by CVT. It uses a shared per-view CNN encoder, cross-view Transformer token
exchange, and U-Net-style image decoder. It is trained on the 2000-object
VG-AlignSeg train split and evaluated on the 187-object test split. The
evaluation restricts predictions to the known actor-id label set for each
object, matching the benchmark's part-label protocol; it does not use GT mask
geometry at inference.

Command:

```bash
PYTHONUNBUFFERED=1 /home/lyx/miniconda3/envs/mova/bin/python \
  scripts/train_cvt_image_adapter.py \
  --output-dir final_results/baselines/cross_view_transformers/image_space_adapter \
  --device cuda:1 \
  --max-steps 1200 \
  --eval-every 300 \
  --eval-max-objects 60 \
  --batch-size 4 \
  --num-workers 4 \
  --amp \
  --viz-samples 40 \
  --log-every 25
```

Training selected the best checkpoint at step 900 by test-subset mIoU.

Output:
`final_results/baselines/cross_view_transformers/image_space_adapter/`

### PartSLIP2 And COPS

The official repositories are checked out at:

- PartSLIP2: `/home/lyx/curriculum/computer_vision/PartSLIP2`
- COPS: `/home/lyx/curriculum/computer_vision/COPS`

Both native methods are 3D part segmentation systems, while the VG-AlignSeg
test split used in this repository is an 8-view 2D rendered-image benchmark.
PartSLIP2/PartSLIP++ requires PartNetE-style point-cloud/projection assets,
category checkpoints, GLIP, and SAM. COPS requires point clouds/meshes and
geometric feature aggregation. Therefore the native repositories are not
directly runnable on the VG-AlignSeg final test set without changing the
benchmark input/output contract.

To still provide visual diagnostic comparisons, I added
`scripts/evaluate_partslip_cops_adapters.py`, which implements two explicitly
labeled 2D proxies:

- `PartSLIP2 2D adapter`: per-view foreground RGB+XY proposal clustering.
- `COPS 2D adapter`: global cross-view foreground RGB+XY+view clustering.

Both adapters predict class-agnostic segment ids from the input RGBA images.
Hungarian matching assigns predicted segment ids to GT actor ids for scoring
and mapped visualization only; no GT mask geometry is copied into predictions.
These results should be reported as task-compatible adapters/proxies, not as
native PartSLIP2 or COPS numbers.

Command:

```bash
PYTHONUNBUFFERED=1 /home/lyx/miniconda3/envs/mova/bin/python \
  scripts/evaluate_partslip_cops_adapters.py \
  --output-dir final_results/baselines/partslip2_cops_visual_adapters \
  --viz-samples 10 \
  --viz-random \
  --viz-seed 20260429 \
  --log-every 10
```

Output:
`final_results/baselines/partslip2_cops_visual_adapters/`

## Final Metrics

| Method | Protocol | iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| --- | --- | ---: | ---: | ---: | ---: |
| VG-AlignSeg V4 | source-guided top-4, strict target-only | 0.5614 | 0.5344 | 0.6171 | 0.9975 |
| CVT image-space adapter | task-compatible CVT modification | 0.2698 | 0.2951 | 0.3207 | 0.9743 |
| COPS 2D adapter | cross-view RGB+XY clustering proxy | 0.2997 | 0.3157 | 0.3449 | 0.9670 |
| PartSLIP2 2D adapter | per-view RGB+XY proposal proxy | 0.2030 | 0.2159 | 0.2383 | 0.9538 |
| PanSt3R | panoptic segment oracle matching | 0.1674 | 0.1853 | 0.2043 | 0.8975 |
| Original Cross View Transformers | native BEV output, not directly applicable | N/A | N/A | N/A | N/A |

## Artifacts

- Combined metrics: `final_results/baselines/metrics_table.json`
- PanSt3R visualizations:
  `final_results/baselines/panst3r/v2_224_oracle_match/visualizations/`
- CVT adapter visualizations:
  `final_results/baselines/cross_view_transformers/image_space_adapter/test_eval/visualizations/`
- CVT adapter training log:
  `final_results/baselines/cross_view_transformers/image_space_adapter/metrics.jsonl`
- PartSLIP2/COPS raw outputs:
  `final_results/baselines/partslip2_cops_visual_adapters/*/raw_outputs/`
- PartSLIP2/COPS individual visualizations:
  `final_results/baselines/partslip2_cops_visual_adapters/*/visualizations/`
- PartSLIP2/COPS paper comparison figures:
  `final_results/baselines/partslip2_cops_visual_adapters/paper_figures/`
