# VG-AlignSeg V2 36845-Views Experiment Report

## Goal

This report verifies a complete VG-AlignSeg V2 pipeline on the local 8-view chair sample:

```text
/home/lyx/curriculum/computer_vision/VG-AlignSeg/36845_views
```

The experiment is intentionally a strict single-object overfit test. Its role is to validate data loading, frozen VGGT feature extraction, cross-view part transfer, high-resolution refinement, loss computation, evaluation, checkpointing, and visualization before scaling to a dataset-level benchmark.

## Model

The runnable V2 model is implemented in:

- `models/vg_alignseg_v2.py`
- `models/highres_prompt_head.py`

The training and evaluation entry points are:

- `scripts/train_v2_36845.py`
- `scripts/evaluate_v2_36845.py`

The final architecture is:

```text
8 RGB views
  |
Frozen VGGT
  |-- token grid
  |-- point grid / confidence
  |
Coarse part head
  |
Geometry-pruned sparse cross-view alignment
  |
Cross-view logit propagation
  |
Cross-view part prototype prompts
  |
Prompted high-resolution refinement head
  |
Per-view part masks
```

## Data

The strict test uses `mask_mode=actors`, which keeps all visible actor masks as separate classes:

```text
0: background
1: actor_2
2: actor_3
3: actor_4
4: actor_5
5: actor_6
6: actor_7
7: actor_8
8: actor_9
9: actor_10
10: actor_11
11: actor_12
12: actor_14
```

This is harder than the semantic 4-class setting because many actors are thin chair-base or wheel components occupying very few pixels.

## Training

Stage 1: train V2 from scratch.

```bash
cd /home/lyx/curriculum/computer_vision/VG-AlignSeg

python scripts/train_v2_36845.py \
  --views 8 \
  --mask-mode actors \
  --steps 800 \
  --lr 2e-4 \
  --coarse-loss-weight 0 \
  --dice-loss-weight 0.5 \
  --consistency-loss-weight 0 \
  --log-every 50 \
  --viz-every 200 \
  --output-dir outputs/v2_36845_actors_proto_800_lr2e4
```

Best stage-1 checkpoint:

```text
outputs/v2_36845_actors_proto_800_lr2e4/v2_36845_best.pt
```

Stage 2: slow convergence from the best checkpoint.

```bash
python scripts/train_v2_36845.py \
  --views 8 \
  --mask-mode actors \
  --steps 500 \
  --lr 3e-5 \
  --coarse-loss-weight 0 \
  --dice-loss-weight 0.5 \
  --consistency-loss-weight 0 \
  --resume outputs/v2_36845_actors_proto_800_lr2e4/v2_36845_best.pt \
  --log-every 50 \
  --viz-every 250 \
  --output-dir outputs/v2_36845_actors_proto_exact_slow
```

## Final Evaluation

Independent evaluation command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_v2_36845.py \
  --checkpoint outputs/v2_36845_actors_proto_exact_slow/v2_36845_best.pt \
  --output-dir outputs/v2_36845_actors_proto_exact_slow/eval
```

Final metrics:

| Metric | Value |
|---|---:|
| pixel accuracy | 1.0000 |
| pixel errors | 0 |
| mean IoU | 1.0000 |
| mean Dice | 1.0000 |
| per-class IoU | 1.0000 for all 13 classes |
| cross-view low-res agreement | 0.9570 |

Final qualitative visualization:

```text
outputs/v2_36845_actors_proto_exact_slow/step_0500.png
outputs/v2_36845_actors_proto_exact_slow/eval/prediction.png
```

## Ablation Observations

Earlier V1/V2 pilot runs established the failure mode and the fix:

| Setting | Result |
|---|---|
| V1 low-res 16x16 upsample, all actors | learns coarse shapes but fails thin actors |
| V1/V2 high-res decoder without pixel memorizer | can approach exact fitting |
| V2 high-res + prototype prompts, no pixel memorizer | reaches exact 8-view actor-mask overfit |
| pixel memorizer diagnostic | reaches exact fit quickly, but is not a deployable model |

The key lesson is that the original V1 cross-view path was not the limiting factor by itself. The main bottleneck was the 16x16 decoder. Once high-resolution refinement is added, even thin base/foot actors become learnable.

## Important Boundary

This is not yet a dataset-level result. It is a complete single-object overfit proof that the architecture can represent the target task. A publishable result still needs:

1. many objects and categories,
2. held-out test objects,
3. real ablations across multiple samples,
4. comparisons against single-view and non-geometric baselines,
5. consistency evaluation under occlusion or missing-view stress tests.
