# Progressive Training Process

The final result was reached through a staged route rather than a single
end-to-end run.

## Stage 1: Direct Multi-Class Segmentation

We first tried direct actor-id segmentation with VGGT features and
high-resolution decoders. This validated the data pipeline, mask rendering, and
overfit capacity, but dataset-level generalization was limited because actor ids
are local to each object and do not define stable semantic categories.

## Stage 2: Source-Guided Binary Part Transfer

V4 reformulated segmentation as binary transfer. For one actor at a time, the
model receives source-view actor masks and predicts that same actor on target
views. The final multi-part segmentation is produced by composing actor logits.

Key modules:

- Frozen VGGT encoder.
- Source-mask prompt encoder.
- Token prototype branch.
- Point-map geometry branch.
- U-Net/DeepLab-style high-resolution decoder.
- Boundary auxiliary loss and iterative refinement.

## Stage 3: Multi-Source Aggregation

Single-source transfer was unstable for occluded or tiny parts. We therefore
aggregated top-k visible source views per actor. Top2 gave the strongest
historical checkpoint, and top4 source prompting at evaluation improved target
coverage.

## Stage 4: Strict Target-Only Evaluation

The final main protocol uses:

- Checkpoint:
  `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`
- Evaluation: `scripts/evaluate_v4_dynamic_target_only.py`
- Source selection: top4 visible source views per actor.
- Source handling: source masks are prompts only.
- Metric views: non-source target views only.
- No source GT copy in the metric.

Final table:

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 56.14 | 53.44 | 61.71 | 99.75 |

## Source-Copy Reference

The old fixedviz result is preserved as an auxiliary reference in
`evaluations/v4_staged_oracle_top4_best4500_fixedviz/`. It copies source GT into
source views and evaluates all eight views, reaching 95.78 part mIoU. This
explains why those visualizations look much stronger, but it is not the strict
target-only metric.

