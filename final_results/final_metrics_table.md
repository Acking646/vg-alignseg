# Final Metrics Table

Protocol: `v4_staged_oracle_top4_best4500_fixedviz`.

This is the staged/oracle protocol used for the final presentation: each actor
uses its top-4 visible source masks, source-view masks are copied into the source
views, and evaluation is performed over all eight views.

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 92.42 | 95.88 | 95.78 | 99.84 |

Raw values:

```json
{
  "iou_object_category": 0.9241670489876199,
  "iou_granularity": 0.9587537201799113,
  "iou_part": 0.9578049078397713,
  "cross_view_consistency_acc": 0.9984204930196511
}
```

Definitions:

- `iou-object-category`: macro average over object categories recovered from
  `luyu1021/vg-alignseg/category_models_list.txt`.
- `iou-granularity`: macro average over actor-count buckets:
  `coarse_1_2_parts`, `medium_3_4_parts`, and `fine_5plus_parts`.
- `iou-part`: full-view segmentation mIoU under the staged/oracle top4 protocol.
- `cross-view consistency acc`: full eight-view pixel accuracy under the same
  protocol.

Important caveat: this is not a no-GT-prompt inference metric. The strict
target-only diagnostic is reported separately in
`evaluations/v4_random2000_top4_dynamic_target_only_top4/summary.json`.

