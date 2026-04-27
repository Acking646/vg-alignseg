# External Baseline Evaluation

External baselines were placed next to this project under
`/home/lyx/curriculum/computer_vision`:

- `panst3r`: official PanSt3R repository.
- `cross_view_transformers`: official Cross View Transformers repository.

The VG-AlignSeg evaluation split is the same as the final target-only report:
the loader scans valid `*_views` objects, uses lexicographic order with no
shuffle, takes the first 2000 objects for training, no validation set, and the
remaining 187 objects for testing.

## Metrics

The four reported metrics follow the final VG-AlignSeg table:

- `iou-object-category`: macro average of object-level mIoU over object
  categories.
- `iou-granularity`: macro average of object-level mIoU over actor-count
  buckets: `coarse_1_2_parts`, `medium_3_4_parts`, and `fine_5plus_parts`.
- `iou-part`: mean object-level part/actor IoU.
- `cross-view consistency acc`: pixel accuracy over all evaluated views.

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
