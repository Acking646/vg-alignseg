# Metrics

This document defines the four metrics used in the VG-AlignSeg result tables
and the external baseline report.

## Notation

For one object `o`:

- `V_o`: the set of 8 rendered views.
- `A_o`: the set of foreground actor/part class ids for this object.
- `Y_o[v, p]`: ground-truth class id at pixel `p` in view `v`.
- `P_o[v, p]`: predicted class id at pixel `p` in view `v`.
- background is class id `0`.

For the strict source-guided VG-AlignSeg V4 protocol, each actor `a` has a
source-view set `S_{o,a}` selected by the top-k visible mask areas. The
evaluated target-view set is:

```text
T_{o,a} = V_o \ S_{o,a}
```

For non-source-guided baselines such as PanSt3R and the CVT image-space
adapter, all 8 views are evaluated.

## Actor/Part IoU

For one object and one actor `a`, the actor IoU is:

```text
IoU(o, a) = |P == a and Y == a| / |P == a or Y == a|
```

The pixel domain is:

- VG-AlignSeg V4 strict target-only: pixels from `T_{o,a}` only.
- PanSt3R diagnostic baseline: all 8 views after Hungarian matching predicted
  panoptic segment ids to GT actor ids.
- CVT image-space adapter: all 8 views.

The object-level part IoU is the mean over actors:

```text
mIoU_object(o) = mean_{a in A_o} IoU(o, a)
```

The table metric `iou-part` is the mean over test objects:

```text
iou-part = mean_{o in test} mIoU_object(o)
```

## iou-object-category

Each test object has an object category, e.g. `Bottle`, `Clock`, `Door`, or
`StorageFurniture`, from `final_results/metadata/object_category_map.json`.

First compute the mean object-level IoU for each category:

```text
IoU_category(c) = mean_{o in test, category(o)=c} mIoU_object(o)
```

Then macro-average over categories:

```text
iou-object-category = mean_c IoU_category(c)
```

This avoids letting categories with many test objects dominate the final score.

## iou-granularity

Each object is assigned to one of three granularity buckets by actor count:

- `coarse_1_2_parts`: 1 or 2 foreground actors.
- `medium_3_4_parts`: 3 or 4 foreground actors.
- `fine_5plus_parts`: 5 or more foreground actors.

First compute the mean object-level IoU for each bucket:

```text
IoU_granularity(g) = mean_{o in test, bucket(o)=g} mIoU_object(o)
```

Then macro-average over the three buckets:

```text
iou-granularity = mean_g IoU_granularity(g)
```

This measures whether the method still works as the number of parts increases.

## Cross-View Consistency Accuracy

The reported `cross-view consistency acc` is a pixel accuracy over the same
evaluation domain used by the method.

For full-view multiclass baselines:

```text
acc = (# pixels where P == Y) / (# evaluated pixels)
```

For strict source-guided VG-AlignSeg V4 target-only evaluation, the evaluator
uses per-actor binary masks on non-source target views:

```text
acc = 1 - (sum_{o,a} # pixels where [P == a] != [Y == a] on T_{o,a})
          / (sum_{o,a} # pixels on T_{o,a})
```

This is why the V4 target-only accuracy is reported together with its protocol
description: source views are excluded, and GT source masks are not copied into
the evaluated pixels.

## Current Reported Metrics

| Method | Protocol | iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| --- | --- | ---: | ---: | ---: | ---: |
| VG-AlignSeg V4 | source-guided top-4, strict target-only | 0.5614 | 0.5344 | 0.6171 | 0.9975 |
| CVT image-space adapter | task-compatible CVT modification | 0.2698 | 0.2951 | 0.3207 | 0.9743 |
| PanSt3R | panoptic segment oracle matching | 0.1674 | 0.1853 | 0.2043 | 0.8975 |
| Original Cross View Transformers | native BEV output, not directly applicable | N/A | N/A | N/A | N/A |

Machine-readable version:
`final_results/baselines/metrics_table.json`.

## Code References

- V4 target-only metric implementation:
  `scripts/evaluate_v4_dynamic_target_only.py::target_only_actor_metrics`
- Category/granularity aggregation:
  `scripts/summarize_target_only_results.py`
- PanSt3R baseline matching and metrics:
  `scripts/evaluate_panst3r_baseline.py`
- CVT image-space adapter metric implementation:
  `scripts/train_cvt_image_adapter.py::actor_metrics`
