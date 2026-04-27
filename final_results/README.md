# VG-AlignSeg Final Results

This directory contains the final runnable result package for the V4 source-guided
part transfer route.

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
target-only evaluation protocol for eight-view part segmentation.

## Final Split

- Dataset root: `data/vg-alignseg-dataset`
- Split seed: `42`
- Random split: yes
- Train samples: `2000`
- Test samples: `187`
- View count: `8`
- Image size: `224`

The manifests are:

- `runs/v4_random2000_top4_independent_seed42/train_manifest.json`
- `runs/v4_random2000_top4_independent_seed42/test_manifest.json`

Both manifests have `shuffle_split=true` and `split_seed=42`, and their object
ids are not monotonically sorted.

## Main Run

Training output:

- `runs/v4_random2000_top4_independent_seed42/`
- Checkpoint used for final strict evaluation: local `best.pt`, step `3000`
  (checkpoint files are intentionally ignored by git).
- Training metrics: `runs/v4_random2000_top4_independent_seed42/metrics.jsonl`
- Training curves:
  - `runs/v4_random2000_top4_independent_seed42/curves/training_loss.png`
  - `runs/v4_random2000_top4_independent_seed42/curves/miou_accuracy.png`
  - `runs/v4_random2000_top4_independent_seed42/curves/pixel_errors.png`
- Training/test snapshots:
  - `runs/v4_random2000_top4_independent_seed42/train_transfer_step_*.png`
  - `runs/v4_random2000_top4_independent_seed42/test_viz_step_003000/`

The in-training test probe was capped at 30 objects. Its best observed mIoU was
`0.7206` at step `3000`, so it should be treated as a quick monitor, not the
final full-test metric.

## Strict Final Evaluation

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

## Upper-Bound Evaluation

For comparison, the earlier top4 fixed visualization run at
`outputs/v4_result_eval_top4_best4500_fixedviz` reported:

- mIoU: `0.9578`
- Pixel accuracy: `0.9984`
- Test objects: `187`

That number is an oracle/upper-bound protocol because the source views are
copied from GT and included in the full 8-view metric. It is useful for showing
that top-k source selection and composition are coherent, but it must not be
reported as the strict no-GT-transfer score.

## Reproduction

See `reproduce_finalresults.sh` for the exact training, curve, and strict
evaluation commands.

