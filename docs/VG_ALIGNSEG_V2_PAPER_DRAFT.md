# VG-AlignSeg: Geometry-Guided Cross-View Prompting for Multi-View Consistent Part Segmentation

## Abstract

Multi-view part segmentation requires both pixel-accurate per-view masks and cross-view consistency of semantic or structural parts. Existing single-view segmentation models can delineate visible regions but do not enforce that the same physical part receives a consistent label across viewpoints. Conversely, direct geometry-based transfer can align views coarsely but often fails near object boundaries and thin structures due to projection drift. We introduce VG-AlignSeg, a geometry-guided multi-view part segmentation framework built on a frozen VGGT backbone. VG-AlignSeg combines three components: geometry-pruned sparse cross-view logit propagation, cross-view part prototype prompts, and a prompted high-resolution refinement head. The sparse propagation branch transfers part evidence only through geometry-plausible token correspondences. The prototype branch aggregates class-wise source-view feature prototypes and injects them as target-view prompts without requiring external query masks at inference time. The high-resolution refinement head fuses VGGT tokens, propagated logits, prototype prompts, RGB features, and positional cues to recover thin part boundaries. On a strict 8-view chair overfit benchmark with 13 actor-level part masks, VG-AlignSeg V2 reaches exact pixel-level fitting with zero pixel errors, validating the representational capacity and end-to-end training pipeline. These results establish a strong prototype for scaling to ShapeNetPart, PartNet, and real multi-view part benchmarks.

## 1. Introduction

Part segmentation is a central problem in 3D-aware visual understanding. For a single image, the goal is to assign each pixel to a semantic part such as chair back, seat, leg, or arm. In multi-view settings, the problem is more demanding: the model must produce accurate masks in every view while preserving a consistent part identity across viewpoint changes, occlusion, and scale variation.

Recent geometry-aware models such as VGGT provide strong multi-view representations by jointly encoding images and predicting camera, depth, point, and tracking-related features. However, directly using point maps or projected correspondences for dense mask transfer is not enough. The VGGT-Segmentor paper observes that raw VGGT point projection can drift at pixel level even when internal cross-view attention remains semantically aligned. This suggests a useful design principle: use geometry and VGGT attention as robust prompts, but let a learned segmentation head perform final dense prediction.

VG-AlignSeg follows this principle for multi-view part segmentation. Instead of treating geometry as a hard warp, we use it to restrict sparse cross-view matching and to form part-aware prompts. The model learns a high-resolution segmentation head that can correct local projection drift and recover thin structures.

## 2. Contributions

The current V2 prototype has four concrete contributions.

1. **Multi-view part segmentation formulation with consistency pressure.** We formulate the task as predicting masks for all views of the same object while sharing evidence across views.

2. **Geometry-pruned sparse part transfer.** We use VGGT point maps and confidence to restrict cross-view token matching to geometry-plausible candidates, then propagate coarse part logits through learned sparse alignment.

3. **Cross-view part prototype prompting.** Inspired by VGGT-Segmentor's mask-prompt fusion, we introduce a no-query variant for multi-view part segmentation. Each view produces class-wise feature prototypes from predicted part probabilities. For a target view, prototypes from other views become source-view part prompts, providing a compact cross-view memory.

4. **Prompted high-resolution refinement.** We remove the original 16x16 bottleneck by decoding at full image resolution from RGB features, VGGT token context, propagated logits, prototype logits, and coordinate cues. This is essential for thin chair-base actors and small parts.

## 3. Related Work

**Multi-view geometry.** Classical multi-view stereo and structure-from-motion recover geometric correspondences but are not designed for semantic part identity. VGGT provides feed-forward multi-view geometric representations and is suitable as a frozen backbone for downstream segmentation.

**Cross-view segmentation.** VGGT-Segmentor targets instance-level ego-exo object correspondence. It shows that VGGT internal feature alignment can remain reliable even when point reprojection drifts, and proposes mask prompt fusion, point-guided prediction, and mask refinement. VG-AlignSeg adapts this lesson to multi-view part segmentation, where the goal is not a single queried object mask but all part masks across all views.

**Part segmentation.** ShapeNetPart and PartNet provide 3D part annotations, but 2D multi-view consistent part segmentation remains less explored. VG-AlignSeg is intended as a bridge between 2D dense masks and 3D-aware multi-view consistency.

## 4. Method

### 4.1 Overview

Given `N` RGB views of the same object,

```text
I = {I_1, ..., I_N}
```

the model predicts per-view part masks:

```text
Y_i in {1, ..., K}^{H x W}.
```

The model consists of:

```text
Frozen VGGT
  -> token grid, point grid, confidence
Coarse Head
  -> per-view low-resolution part logits
Geometry-Pruned Sparse Alignment
  -> cross-view propagated logits
Cross-View Prototype Prompt
  -> source-view part prototype logits
Prompted High-Resolution Refinement
  -> final full-resolution part masks
```

### 4.2 Frozen VGGT Backbone

VGGT processes all views jointly. We use the frozen aggregator output as token features:

```text
T in R^{B x N x 16 x 16 x C}
```

and downsample VGGT point maps into token-level geometry:

```text
X in R^{B x N x 16 x 16 x 3}, C_geo in R^{B x N x 16 x 16}.
```

The backbone is frozen to reduce memory and to focus learning on cross-view segmentation heads.

### 4.3 Coarse Part Head

A lightweight convolutional head predicts per-view low-resolution logits:

```text
L_i = CoarseHead(T_i).
```

These logits are not expected to be pixel-accurate; they provide initial part distributions for transfer and prompting.

### 4.4 Geometry-Pruned Sparse Logit Propagation

For every target-source view pair `(i, j)`, we compute candidate source tokens for each target token using VGGT token-level 3D distance. Low-confidence geometry is masked, and only top-k candidates are kept.

Alignment scores combine feature similarity and geometric proximity:

```text
s(p, q) = alpha * cos(T_i^p, T_j^q) - beta * ||X_i^p - X_j^q||_2.
```

Softmax over the top-k candidates gives sparse alignment weights. Source logits are propagated to the target:

```text
M_i = mean_j A_ij L_j.
```

This gives a geometry-aware cross-view part prior.

### 4.5 Cross-View Part Prototype Prompt

VGGT-Segmentor uses a source mask as an explicit prompt. In our all-view part segmentation setting, source masks are not externally provided at inference time. We therefore build prompts from the model's own coarse part probabilities.

For each view and part class, we compute a weighted feature prototype:

```text
P_{i,k} = sum_p softmax(L_i)_{p,k} normalize(T_i^p)
          / sum_p softmax(L_i)_{p,k}.
```

For a target view, prototypes from all other views are averaged and compared with target tokens:

```text
R_{i,p,k} = tau * cos(T_i^p, mean_{j != i} P_{j,k}).
```

`R` is a compact class-wise prompt map. It acts as a multi-view part memory and encourages part identity consistency across viewpoints.

### 4.6 Prompted High-Resolution Refinement

The final decoder receives:

```text
target VGGT tokens
target coarse logits
cross-view propagated logits
cross-view prototype prompt logits
RGB image features
2D coordinates
```

The low-resolution token stream is fused and upsampled to full resolution. RGB features and coordinates are concatenated before several residual convolutional blocks predict final logits:

```text
Y_i = Refine(T_i, L_i, M_i, R_i, I_i).
```

This is the key component that fixes the original 16x16 bottleneck and recovers thin actor masks.

### 4.7 Training Loss

For the current strict overfit setting, the effective loss is:

```text
L = CE(Y, Y*) + lambda_dice Dice(Y, Y*).
```

The code also supports auxiliary coarse loss and consistency loss:

```text
L_total = L_final
        + lambda_coarse L_coarse
        + lambda_cons KL(Y_low || M).
```

In pilot experiments, applying consistency too early destabilized single-sample optimization. The recommended schedule is two-stage:

1. train final high-resolution masks with CE + Dice,
2. add small consistency regularization only after segmentation stabilizes.

## 5. Experiments

### 5.1 Dataset

We use the local 8-view chair sample:

```text
36845_views/
  color/
  depth/
  part_mask/actor_*/
  point_cloud/
```

The strict actor-level setting uses 13 classes: background plus 12 actor masks. This is harder than semantic chair-part segmentation because many actors are tiny wheel or base components.

### 5.2 Training Protocol

Stage 1:

```bash
python scripts/train_v2_36845.py \
  --views 8 \
  --mask-mode actors \
  --steps 800 \
  --lr 2e-4 \
  --coarse-loss-weight 0 \
  --dice-loss-weight 0.5 \
  --consistency-loss-weight 0 \
  --output-dir outputs/v2_36845_actors_proto_800_lr2e4
```

Stage 2:

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
  --output-dir outputs/v2_36845_actors_proto_exact_slow
```

### 5.3 Results

Final independent evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_v2_36845.py \
  --checkpoint outputs/v2_36845_actors_proto_exact_slow/v2_36845_best.pt \
  --output-dir outputs/v2_36845_actors_proto_exact_slow/eval
```

| Metric | Result |
|---|---:|
| Pixel accuracy | 1.0000 |
| Pixel errors | 0 |
| Mean IoU | 1.0000 |
| Mean Dice | 1.0000 |
| Per-class IoU | 1.0000 for all 13 classes |
| Cross-view low-res agreement | 0.9570 |

The exact-fit visualization is stored at:

```text
outputs/v2_36845_actors_proto_exact_slow/step_0500.png
outputs/v2_36845_actors_proto_exact_slow/eval/prediction.png
```

### 5.4 Ablation Plan

The current local experiments support the following ablation table structure:

| Variant | Expected role |
|---|---|
| V1 low-res decoder | shows 16x16 bottleneck failure |
| High-res decoder only | tests whether pixel detail is recoverable |
| + sparse geometry propagation | tests geometry-aware cross-view prior |
| + prototype prompts | tests cross-view part identity memory |
| + consistency loss after warmup | tests explicit view-consistency regularization |

For a paper, these should be run over many objects, not just one overfit sample.

## 6. Discussion

The main technical lesson is that cross-view part segmentation should not rely on hard geometric projection alone. VGGT point maps and attention are powerful, but dense boundaries require a learned high-resolution decoder. Our part prototype prompts provide an efficient way to transfer identity information across views without external query masks.

The current 8-view exact-overfit result validates model capacity and code correctness. It does not yet prove generalization. The next milestone is to construct a multi-object benchmark from ShapeNetPart or PartNet and report held-out mIoU, cross-view consistency, and robustness to occlusion.

## 7. Limitations and Next Steps

Current limitations:

1. only a single local 8-view sample has been fully validated,
2. actor masks are instance-like and not always semantic parts,
3. consistency loss needs a warmup schedule,
4. prototype prompts are currently derived from predicted coarse logits, not from learned prompt tokens,
5. no comparison to SAM/SAM2/Mask2Former baselines yet.

Next steps for a true submission:

1. generate a ShapeNetPart/PartNet multi-view benchmark,
2. train on multiple categories and test held-out objects,
3. add occlusion/missing-view stress tests,
4. compare single-view high-res, geometry-only, prototype-only, and full V2,
5. report runtime and memory against VGGT-Segmentor-style baselines.

## 8. Suggested Paper Positioning

The strongest title direction is:

```text
VG-AlignSeg: Cross-View Prototype Prompting for Geometry-Consistent Multi-View Part Segmentation
```

The most defensible novelty claim is not "we invented cross-view segmentation"; it is:

```text
We adapt VGGT-style cross-view geometry to all-view part segmentation by replacing external source-mask queries with self-generated cross-view part prototype prompts, and by combining sparse geometry-pruned logit transfer with high-resolution boundary refinement.
```

This is meaningfully different from VGGT-Segmentor, which solves source-mask queried ego-exo instance correspondence. VG-AlignSeg instead solves simultaneous multi-view part labeling without an external query mask.
