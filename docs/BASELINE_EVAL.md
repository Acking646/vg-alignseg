# External Baseline Evaluation

External baselines were placed next to this project under
`/home/lyx/curriculum/computer_vision`:

- `panst3r`: official PanSt3R repository.
- `cross_view_transformers`: official Cross View Transformers repository.
- `PartSLIP2`: official PartSLIP2/PartSLIP++ repository.
- `COPS`: official COPS repository.

The VG-AlignSeg evaluation split is the same as the final target-only report:
the loader scans valid `*_views` objects, uses lexicographic order with no
shuffle, takes the first 2000 objects for training, no validation set, and the
remaining 187 objects for testing.

## Metrics

The four reported metrics follow the final VG-AlignSeg table:

- `iou-object-category`: category-balanced macro average of object-level
  actor/part IoU.
- `iou-granularity`: granularity-balanced macro average over actor-count
  buckets: `coarse_1_2_parts`, `medium_3_4_parts`, and `fine_5plus_parts`.
- `iou-part`: mean object-level actor/part IoU.
- `cross-view consistency acc`: pixel accuracy over the evaluated pixel domain.
  For source-guided V4 this is actor-wise binary accuracy on non-source target
  views; for full-view baselines it is multiclass pixel accuracy over all 8
  views.

See `docs/METRICS.md` for exact formulas.

## PanSt3R Protocol

PanSt3R produces category-level panoptic segment ids, not VG-AlignSeg actor ids.
For a fair but explicit diagnostic baseline, the evaluator runs PanSt3R on the
same 8 RGB views and then performs class-agnostic Hungarian matching from
predicted panoptic segments to GT actor masks. This matching assigns names only
for scoring; it does not copy source/target GT masks into the prediction.

Output:
`final_results/baselines/panst3r/v2_224_oracle_match/`

## Cross View Transformers Protocol

Cross View Transformers is not directly applicable to this benchmark in its
original form. Its model expects vehicle-camera datasets with calibration and
BEV map labels, and it returns BEV semantic maps. VG-AlignSeg requires
object-centric image-space part/actor masks.

For a reasonable task-compatible comparison, this project adds a small
image-space adapter inspired by CVT:

- shared per-view CNN encoder
- cross-view Transformer token exchange
- U-Net-style image decoder

The adapter is trained on the VG-AlignSeg train split and evaluated on the same
187-object test split. See
`final_results/baselines/cross_view_transformers/image_space_adapter/`.

## PartSLIP2 And COPS Protocol

The native PartSLIP2 and COPS repositories are not directly applicable to the
current VG-AlignSeg final test split because they are 3D part segmentation
pipelines. PartSLIP2/PartSLIP++ expects point-cloud/projection assets and
external GLIP/SAM/category checkpoints; COPS expects point clouds/meshes and
geometric feature aggregation. The VG-AlignSeg split used here contains 8-view
2D RGBA renderings and 2D actor masks.

For visual and quantitative diagnostics, this project therefore reports
explicit 2D adapters:

- `PartSLIP2 2D adapter`: per-view foreground RGB+XY proposal clustering.
- `COPS 2D adapter`: global cross-view foreground RGB+XY+view clustering.

Both adapters predict class-agnostic segment ids. Hungarian matching assigns
segment ids to GT actor ids for evaluation and mapped visualization only. No GT
mask is copied into the prediction. These should be described as proxies, not
native PartSLIP2 or COPS results.

Output:
`final_results/baselines/partslip2_cops_visual_adapters/`

The folder contains per-method raw outputs, per-method visualization sheets,
combined paper-ready comparison figures, and native-status JSON files explaining
why the original repositories are not directly runnable on the 2D benchmark.
