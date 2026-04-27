# VG-AlignSeg Final Results

This directory contains the cleaned result package for the V4 source-guided part
transfer route. The main reported result is the strict target-only top4
evaluation generated from the stronger historical V4 checkpoint.

## Method Summary

V4 is source-guided binary part transfer rather than global actor-id
classification.

- Input: eight RGB views of one object, one actor mask from one or more source
  views, and the source view ids.
- Output: the same actor's binary mask on target views.
- Final segmentation: run the transfer head once per actor and compose actor
  logits into a multi-part mask.

VGGT/VGGT-S motivates the frozen multi-view geometric backbone, source-mask
prompting, point-map cues, and refinement idea. This project adapts those ideas
to eight-view object-centric local actor labels with top-k multi-source
aggregation and explicit target-only evaluation.

## Main Strict Target-Only Result

Primary output:

- `evaluations/v4_target_only_top4_best4500/`
- Visualizations:
  `evaluations/v4_target_only_top4_best4500/visualizations/`
- Metrics: `evaluations/v4_target_only_top4_best4500/summary.json`
- Per-object/category/granularity tables:
  `evaluations/v4_target_only_top4_best4500/per_object_enriched.json`,
  `per_category.json`, and `per_granularity.json`

Protocol:

- For every actor, select the top-4 visible source views by GT mask area.
- Use source masks as prompts only.
- Evaluate only non-source target views.
- Do not copy source GT into the prediction or metric.
- Compose actor transfer logits with max merging and threshold `-0.5`.

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 56.14 | 53.44 | 61.71 | 99.75 |

Raw values are stored in `final_metrics_table.md` and in the evaluation
`summary.json`.

## Auxiliary Source-Copy Reference

The directory `evaluations/v4_staged_oracle_top4_best4500_fixedviz/` preserves
the earlier strong fixed visualization setting:

- same checkpoint and top4 source selection;
- source-view GT masks are copied into source views;
- the metric is computed over all eight views.

This reference reaches 92.42 category IoU, 95.88 granularity IoU, 95.78 part
mIoU, and 99.84 pixel accuracy. It is useful for visualizing the upper guided
composition behavior, but the strict target-only result above is the main
evaluation.

## Dataset Categories

Object-category metadata is taken from the Hugging Face dataset repository
`luyu1021/vg-alignseg`, file `category_models_list.txt`. The repository lists
46 object categories and 2339 object ids. Local parsed metadata is in:

- `metadata/category_models_list.txt`
- `metadata/object_category_map.json`
- `metadata/category_summary.json`

The test split used by the final evaluation covers 12 categories:

- Bottle, Box, Chair, Clock, Display, Door, Faucet, Keyboard, Laptop,
  Microwave, Oven, StorageFurniture.

## Final Split

- Dataset root: `data/vg-alignseg-dataset`
- Split: continuous 2000/187 train/test split used by the best historical V4
  checkpoint and fixedviz result.
- Split policy: lexicographic object-directory order, no shuffle.
- Total valid objects: `2187`
- Train samples: `2000`
- Test samples: `187`
- View count: `8`
- Image size: `224`

The strict target-only manifest is:

- `evaluations/v4_target_only_top4_best4500/test_manifest.json`

The full dataset-source and split record is:

- `../docs/DATASET_SPLIT.md`
- `split_summary.json`

## Training Artifacts

The final package keeps lightweight artifacts from the strong run in:

- `training/v4_top2_phase2_random_copy/metrics.jsonl`
- `training/v4_top2_phase2_random_copy/run_config.json`
- `training/v4_top2_phase2_random_copy/curves/training_loss.png`
- `training/v4_top2_phase2_random_copy/curves/miou_accuracy.png`
- `training/v4_top2_phase2_random_copy/curves/pixel_errors.png`

Checkpoint files are intentionally ignored by git. The strict evaluation expects
the checkpoint at:

- `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`

## Reproduction

See `reproduce_finalresults.sh` for the exact target-only evaluation,
category/granularity summary, optional source-copy reference evaluation, and
curve plotting commands.

Training commands are documented in `../docs/TRAINING.md`. The convenience
launcher is `../scripts/train_v4_final.sh`.
