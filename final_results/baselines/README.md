# Baseline Evaluation Notes

This directory stores external baseline checks on the VG-AlignSeg test split
(`train_count=2000`, `val_count=0`, remaining `187` objects as test).

Metric definitions and formulas are documented in `docs/METRICS.md`.

## VG-AlignSeg V4 Target-Only Reference

Source: `final_results/evaluations/v4_target_only_top4_best4500/summary.json`

| Method | Protocol | iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| --- | --- | ---: | ---: | ---: | ---: |
| VG-AlignSeg V4 | source-guided top-4, strict target-only | 0.5614 | 0.5344 | 0.6171 | 0.9975 |
| CVT image-space adapter | task-compatible CVT modification, trained on VG-AlignSeg | 0.2698 | 0.2951 | 0.3207 | 0.9743 |
| PanSt3R | class-agnostic panoptic segments, Hungarian oracle matched to GT actor ids | 0.1674 | 0.1853 | 0.2043 | 0.8975 |
| Original Cross View Transformers | not directly applicable; BEV map segmentation output | N/A | N/A | N/A | N/A |

## PanSt3R

Repository: `/home/lyx/curriculum/computer_vision/panst3r`

Checkpoint:
`/home/lyx/curriculum/computer_vision/baseline_weights/panst3r/panst3r_v2_512_5ds.pth`

Evaluation output:
`final_results/baselines/panst3r/v2_224_oracle_match/`

PanSt3R predicts multi-view panoptic segments with category prompts. It does
not predict ShapeNetPart actor ids, so the reported values use Hungarian
matching from predicted segment ids to GT actors for evaluation only. No GT mask
is copied into the prediction.

Run command:

```bash
PYTHONUNBUFFERED=1 /home/lyx/miniconda3/envs/panst3r_eval/bin/python \
  scripts/evaluate_panst3r_baseline.py \
  --output-dir final_results/baselines/panst3r/v2_224_oracle_match \
  --device cuda:0 \
  --image-size 224 \
  --num-keyframes 8 \
  --max-bs 1 \
  --amp bf16 \
  --log-every 10 \
  --viz-samples 40
```

## Cross View Transformers

Repository: `/home/lyx/curriculum/computer_vision/cross_view_transformers`

Cross View Transformers is designed for nuScenes/Argoverse vehicle-camera
inputs and outputs BEV map-view semantic segmentation (`pred["bev"]`). VG-
AlignSeg requires object-centric 8-view image-space part masks with actor ids.
Because the input geometry, supervision, output space, and labels are different,
the repository cannot be evaluated on the VG-AlignSeg test split without
designing and training a new adapter/model.

For a reasonable task-compatible comparison, `scripts/train_cvt_image_adapter.py`
implements a small image-space adapter inspired by CVT: a shared per-view CNN
encoder, cross-view Transformer token exchange, and U-Net-style image decoder.
It is trained on the VG-AlignSeg train split and evaluated on the same held-out
187-object test split.

Adapter output:
`final_results/baselines/cross_view_transformers/image_space_adapter/`

Run command:

```bash
PYTHONUNBUFFERED=1 /home/lyx/miniconda3/envs/mova/bin/python \
  scripts/train_cvt_image_adapter.py \
  --output-dir final_results/baselines/cross_view_transformers/image_space_adapter \
  --device cuda:1 \
  --max-steps 1200 \
  --eval-every 300 \
  --eval-max-objects 60 \
  --batch-size 4 \
  --num-workers 4 \
  --amp \
  --viz-samples 40 \
  --log-every 25
```
