# Final Metrics Table

Main protocol: `v4_target_only_top4_best4500`.

For each actor, the evaluator chooses its top-4 visible source views by mask
area, uses those source masks as prompts, predicts the remaining target views,
and evaluates only non-source target views. Source GT is not copied into the
metric.

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 56.14 | 53.44 | 61.71 | 99.75 |

Raw values:

```json
{
  "iou_object_category": 0.5614304639185613,
  "iou_granularity": 0.5343687080265572,
  "iou_part": 0.6170851281547065,
  "cross_view_consistency_acc": 0.9974540452927889
}
```

Definitions:

- `iou-object-category`: macro average over object categories recovered from
  `luyu1021/vg-alignseg/category_models_list.txt`.
- `iou-granularity`: macro average over actor-count buckets:
  `coarse_1_2_parts`, `medium_3_4_parts`, and `fine_5plus_parts`.
- `iou-part`: actor-wise IoU on non-source target views only.
- `cross-view consistency acc`: pixel accuracy over the same target-only actor
  masks.

Auxiliary reference:

`evaluations/v4_staged_oracle_top4_best4500_fixedviz/summary.json` reports the
source-copy full-view reference result: 92.42 category IoU, 95.88 granularity
IoU, 95.78 part mIoU, and 99.84 pixel accuracy. It copies source masks into
source views and is not the main strict target-only metric.

