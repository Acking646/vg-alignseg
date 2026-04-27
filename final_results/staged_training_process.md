# Progressive Training Process

This project reached the final top4 result through a staged route rather than a
single end-to-end run.

## Stage 1: Direct Multi-Class Segmentation

We first tried direct actor-id segmentation with VGGT features and high-resolution
decoders. This validated data loading, mask rendering, and overfit capacity, but
dataset-level generalization was limited because actor ids are local to each
object and do not define stable semantic categories.

## Stage 2: Source-Guided Binary Part Transfer

V4 reformulated segmentation as binary transfer. For one actor at a time, the
model receives a source-view actor mask and predicts that same actor in all
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
aggregated top-k visible source views per actor. Top2 improved the full-view copy
metric to the low 80s. Top4 produced the best staged/oracle result.

## Stage 4: Final Staged/Oracle Evaluation

The final presentation protocol uses:

- Checkpoint: `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`
- Evaluation: `scripts/evaluate_v4_staged_oracle.py`
- Source selection: top4 visible source views per actor.
- Source handling: source-view masks are copied into source views.
- Metric views: all eight views.

Final table:

| iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| ---: | ---: | ---: | ---: |
| 92.42 | 95.88 | 95.78 | 99.84 |

## Strict Diagnostic

The strict target-only protocol does not copy source GT into the metric and only
evaluates non-source target views. It reaches 46.02 actor mIoU. This gap explains
why the final staged/oracle visualization is much stronger than the strict
diagnostic visualization.

