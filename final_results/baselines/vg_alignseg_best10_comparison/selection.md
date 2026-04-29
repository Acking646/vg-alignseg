# VG-AlignSeg Best-10 Comparison Selection

This folder contains paper comparison figures for the 10 test objects where
VG-AlignSeg V4 obtains the highest object-level part mIoU in
`final_results/evaluations/v4_target_only_top4_best4500/per_object_enriched.json`.

The figures compare:

- RGB input
- GT mask overlay
- VG-AlignSeg V4 prediction
- PartSLIP2 2D adapter prediction
- COPS 2D adapter prediction

Selection command:

```bash
PYTHONUNBUFFERED=1 /home/lyx/miniconda3/envs/mova/bin/python \
  scripts/evaluate_partslip_cops_adapters.py \
  --output-dir final_results/baselines/vg_alignseg_best10_comparison \
  --viz-indices 78,90,89,83,104,96,105,80,97,103 \
  --viz-samples 10 \
  --log-every 20
```

Selected objects:

| index | object_id | category | actors | VG-AlignSeg V4 mIoU |
| ---: | --- | --- | ---: | ---: |
| 78 | 6500 | Clock | 1 | 0.994821 |
| 90 | 6813 | Clock | 1 | 0.994386 |
| 89 | 6808 | Clock | 1 | 0.994327 |
| 83 | 6641 | Clock | 1 | 0.993958 |
| 104 | 7064 | Clock | 1 | 0.993927 |
| 96 | 6953 | Clock | 1 | 0.992496 |
| 105 | 7068 | Clock | 1 | 0.991646 |
| 80 | 6608 | Clock | 1 | 0.990994 |
| 97 | 6963 | Clock | 1 | 0.990661 |
| 103 | 7054 | Clock | 1 | 0.989630 |

Note: this is intentionally a best-case qualitative selection for VG-AlignSeg.
It is not the aggregate benchmark table. The full-test metrics remain in
`final_results/baselines/metrics_table.json`.
