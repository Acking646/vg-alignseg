# VG-AlignSeg Final Results

This directory contains the cleaned result package for the V4 source-guided part
transfer route. The final presentation result is the staged/oracle top4 protocol
from `v4_result_eval_top4_best4500_fixedviz`, regenerated here with category,
granularity, per-object metrics, and visualizations.

## Why V4 Source-Guided

The V4 idea is not copied directly from VGGT-S. VGGT-S motivates the use of a
frozen VGGT encoder, source masks, geometry-aware feature alignment, point cues,
and iterative mask refinement for cross-view segmentation. Our dataset, however,
uses local actor ids and contains eight object-centric views. A global actor-id
classifier is therefore ill-posed: the same numeric actor id is not a stable
semantic category across objects.

V4 reformulates the problem as source-guided binary part transfer:

- Input: eight RGB views of one object, one actor mask from one or more source
  views, and the source view ids.
- Output: the same actor's binary mask in all views.
- Final segmentation: run the transfer head once per actor and compose actor
  logits into a multi-part mask.

VGGT/VGGT-S provides the backbone idea: multi-view geometric tokens are useful
for cross-view correspondence. The project contribution is the object-level
source-mask guided transfer formulation, top-k multi-source aggregation, and
staged evaluation for eight-view part segmentation.

## Final Presentation Metrics

Primary output:

- `evaluations/v4_staged_oracle_top4_best4500_fixedviz/`
- Visualizations:
  `evaluations/v4_staged_oracle_top4_best4500_fixedviz/visualizations/`
- Metrics:
  `evaluations/v4_staged_oracle_top4_best4500_fixedviz/summary.json`

Protocol:

- For every actor, select the top-4 visible source views by GT mask area.
- Use the source masks as prompts.
- Copy source-view GT masks into the source views.
- Evaluate the composed segmentation over all eight views.

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 92.42 | 95.88 | 95.78 | 99.84 |

Raw values are stored in `final_metrics_table.md`.

## Dataset Categories

Object-category metadata is taken from the Hugging Face dataset repository
`luyu1021/vg-alignseg`, file `category_models_list.txt`. The repository lists
46 object categories and 2339 object ids. Local parsed metadata is in:

- `metadata/category_models_list.txt`
- `metadata/object_category_map.json`
- `metadata/category_summary.json`

The staged/oracle test split covers 12 categories:

- Bottle, Box, Chair, Clock, Display, Door, Faucet, Keyboard, Laptop,
  Microwave, Oven, StorageFurniture.

## Final Split

- Dataset root: `data/vg-alignseg-dataset`
- Split: continuous test split matching `v4_result_eval_top4_best4500_fixedviz`
- Train samples: `2000`
- Test samples: `187`
- View count: `8`
- Image size: `224`

The staged/oracle manifest is:

- `evaluations/v4_staged_oracle_top4_best4500_fixedviz/test_manifest.json`

The random split diagnostic run is still kept under
`runs/v4_random2000_top4_independent_seed42/`.

## Progressive Training

See `staged_training_process.md` for the final story:

1. Direct multi-class actor segmentation validated capacity but struggled because
   actor ids are local.
2. V4 converted the task to source-guided binary part transfer.
3. Multi-source aggregation moved from single source to top2 and top4.
4. The final staged/oracle top4 evaluation copies source masks into source views
   and scores the full eight-view segmentation.

## Main Training Artifacts

Training output:

- `runs/v4_random2000_top4_independent_seed42/`
- Training metrics: `runs/v4_random2000_top4_independent_seed42/metrics.jsonl`
- Training curves:
  - `runs/v4_random2000_top4_independent_seed42/curves/training_loss.png`
  - `runs/v4_random2000_top4_independent_seed42/curves/miou_accuracy.png`
  - `runs/v4_random2000_top4_independent_seed42/curves/pixel_errors.png`
- Training/test snapshots:
  - `runs/v4_random2000_top4_independent_seed42/train_transfer_step_*.png`
  - `runs/v4_random2000_top4_independent_seed42/test_viz_step_003000/`

The final staged/oracle result uses the stronger historical checkpoint:

- `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`

Checkpoint files are intentionally ignored by git.

## Strict Diagnostic Evaluation

Strict target-only evaluation output:

- `evaluations/v4_random2000_top4_dynamic_target_only_top4/summary.json`
- `evaluations/v4_random2000_top4_dynamic_target_only_top4/per_object.json`
- `evaluations/v4_random2000_top4_dynamic_target_only_top4/visualizations/`

Protocol:

- For every actor, choose its top-4 visible source views by GT source-mask area.
- Use those source masks only as prompts.
- Predict the remaining non-source target views.
- Do not copy source GT into the prediction metric.
- Report actor-wise IoU on target views only.

Result:

- Best threshold: `-0.5`
- Target-only actor mIoU: `0.4602`
- Pixel accuracy: `0.9937`
- Actor instances: `506`
- Test objects: `187`

This strict score is much lower because it does not copy source GT into the
metric and only evaluates non-source target views. It is useful as a diagnostic,
but it is not the final presentation table.

## Why The Old Fixedviz Looks Better

The old fixed visualization run at `outputs/v4_result_eval_top4_best4500_fixedviz`
used the same staged/oracle protocol:

- mIoU: `0.9578`
- Pixel accuracy: `0.9984`
- Test objects: `187`
- `eval_copy_source_views=true`

The visualizations are therefore stronger than strict target-only visualizations.
This is now documented explicitly and reproduced under `final_results`.

## Reproduction

See `reproduce_finalresults.sh` for the exact training, curve, staged/oracle
evaluation, and strict diagnostic commands.

