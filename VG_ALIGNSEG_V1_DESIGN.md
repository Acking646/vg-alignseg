# VG-AlignSeg V1 Design

## Goal

Build a first practical version of VG-AlignSeg for multi-view part segmentation under the following constraints:

- Hardware: `2 x RTX 3090 24GB`
- Backbone: `VGGT frozen`
- View count: `4`
- Input size: `224 x 224`
- Matching space: `16 x 16` token grid
- Training focus: only train the new segmentation, alignment, and refinement modules

The main idea is:

1. use frozen VGGT to provide geometry and view-aware features,
2. predict coarse per-view part logits,
3. align views with geometry-pruned sparse matching,
4. refine each target view with aggregated cross-view evidence.

## V1 Summary

The recommended first version is:

- `Frozen VGGT`
- `Low-resolution token alignment only`
- `Geometry top-k pruning`
- `Logits-only propagation`
- `Light refinement head`
- `Two-stage training`

This keeps the method clean and makes it realistic on commodity GPUs.

## Model Overview

```text
Input views {I1 ... IN}
    |
    v
Frozen VGGT
    |- token features Ti
    |- world points Xi
    |- geometry confidence Ci
    |
    +--> CoarseHead(Ti) -> Li
    |
    +--> GeometryPruner(Xi, Xj, Ci, Cj) -> candidate set Nij
    |
    +--> SparseAlign(Ti, Tj, Xi, Xj, Nij) -> Aij
    |
    +--> Propagate logits Aij * Lj -> Mij
    |
    +--> Aggregate over source views -> Mi
    |
    +--> RefineHead([Ti, Li, Mi]) -> Yi
```

## Tensor Shapes

Assume:

- batch size `B`
- number of views `N = 4`
- image size `224 x 224`
- VGGT patch size `14`
- token grid `16 x 16`
- patch token count `P = 256`
- token dim `C = 2048`
- part class count `K`

### Backbone outputs

From `aggregator(images)`:

- input images: `[B, N, 3, 224, 224]`
- aggregated token list element: `[B, N, 261, 2048]`

Reason:

- `1` camera token
- `4` register tokens
- `256` patch tokens
- total `261`

Note:

- VGGT's `Aggregator` concatenates frame-attention and global-attention features along the channel dimension, so the usable token feature dimension is `2 x 1024 = 2048`.

Patch tokens only:

- `tokens = x[:, :, 5:, :]`
- shape: `[B, N, 256, 1024]`
- reshape to token grid: `[B, N, 16, 16, 1024]`

Dense geometry from VGGT heads:

- point map: `[B, N, 224, 224, 3]`
- point confidence: `[B, N, 224, 224]`
- depth map: `[B, N, 224, 224, 1]`
- depth confidence: `[B, N, 224, 224]`

For alignment, downsample geometry to the token grid:

- tokenized point map: `[B, N, 16, 16, 3]`
- tokenized geometry confidence: `[B, N, 16, 16]`

### Trainable modules

Coarse head:

- input: `[B, N, 16, 16, 1024]`
- output coarse logits: `[B, N, 16, 16, K]`

Sparse alignment from source view `j` to target view `i`:

- target tokens: `[B, 16, 16, 1024]`
- source tokens: `[B, 16, 16, 1024]`
- flattened as `[B, 256, 1024]`
- geometry-pruned candidate indices: `[B, 256, topk]`
- alignment weights: `[B, 256, topk]`

Transferred message:

- source coarse logits: `[B, 256, K]`
- propagated logits from one source: `[B, 256, K]`
- aggregated over `N - 1` source views: `[B, 256, K]`
- reshape to grid: `[B, 16, 16, K]`

Refinement input:

- target token features: `[B, 16, 16, 1024]`
- target coarse logits: `[B, 16, 16, K]`
- aggregated message: `[B, 16, 16, K]`
- concat input: `[B, 16, 16, 1024 + 2K]`

Refinement output:

- low-res final logits: `[B, 16, 16, K]`
- upsampled final logits: `[B, K, 224, 224]`

## Module List

### 1. `FrozenVGGTBackbone`

Responsibilities:

- run VGGT forward without gradient
- expose token features for segmentation
- expose geometry outputs for cross-view pruning

Suggested output dict:

```python
{
    "token_grid": [B, N, 16, 16, 1024],
    "point_map": [B, N, 224, 224, 3],
    "point_conf": [B, N, 224, 224],
    "depth": [B, N, 224, 224, 1],
    "depth_conf": [B, N, 224, 224],
}
```

Implementation notes:

- wrap VGGT loading in one module
- freeze all VGGT parameters
- use `torch.no_grad()` in forward
- downsample geometry once inside the wrapper to avoid repeated compute

### 2. `CoarseHead`

Responsibilities:

- predict initial part logits for each view independently

Minimal V1 design:

- `1x1 conv` projection from `1024 -> 256`
- `3x3 conv`
- `GroupNorm`
- `ReLU`
- `3x3 conv`
- `1x1 classifier -> K`

Why this is enough:

- the goal of V1 coarse logits is to seed refinement
- the research novelty is not the single-view head itself

### 3. `GeometryPruner`

Responsibilities:

- avoid full all-pairs token matching
- use 3D geometry to select only promising source candidates

Suggested inputs:

- target token points `X_i`: `[B, 256, 3]`
- source token points `X_j`: `[B, 256, 3]`
- target confidence `C_i`: `[B, 256]`
- source confidence `C_j`: `[B, 256]`

Suggested output:

- `candidate_idx`: `[B, 256, topk]`
- `candidate_mask`: `[B, 256, topk]`

V1 rule:

- compute pairwise 3D distance between target tokens and source tokens
- mask low-confidence geometry
- keep `topk = 8` nearest source tokens per target token

Optional additions later:

- visibility gating
- camera reprojection gating
- bidirectional pruning

### 4. `SparseAlign`

Responsibilities:

- compute alignment only on geometry-pruned candidates

V1 scoring:

```text
s_ij(p, q) = alpha * cosine(f_i^p, f_j^q) - beta * ||x_i^p - x_j^q||_2
```

where:

- `f_i^p` is the target token feature
- `f_j^q` is the source token feature
- `x_i^p` and `x_j^q` are token-level 3D points

Then:

- apply softmax over the `topk` source candidates
- get sparse alignment weights `A_ij`

V1 defaults:

- `topk = 8`
- `alpha = 1.0`
- `beta = 5.0`

### 5. `LogitPropagator`

Responsibilities:

- propagate only coarse logits from source views to the target view

For one target view `i`:

```text
M_ij = A_ij * L_j
M_i = mean_{j != i}(M_ij)
```

Why logits-only first:

- lighter than feature propagation
- easier to stabilize
- easier to interpret in ablations

### 6. `RefineHead`

Responsibilities:

- combine target-view information with propagated cross-view evidence

Input:

- target tokens
- target coarse logits
- aggregated propagated logits

V1 design:

- project token features `1024 -> 256`
- concat with `Li` and `Mi`
- `2` residual conv blocks
- `1` classifier head
- bilinear upsample to `224 x 224`

This is intentionally lightweight. V1 should not depend on a heavy decoder.

## Forward Pass

For each batch:

1. run frozen VGGT on all views
2. extract token grid `T`
3. predict coarse logits `L`
4. for each target view `i`:
5. for each source view `j != i`:
6. run geometry pruning `Nij`
7. run sparse alignment `Aij`
8. propagate source logits to target `Mij`
9. aggregate messages `Mi`
10. refine target output `Yi`
11. compute losses on coarse and final predictions

## Loss Design

V1 total loss:

```text
L = L_final + lambda_coarse * L_coarse + lambda_cons * L_cons
```

### `L_final`

Main segmentation loss on final prediction:

- cross entropy, or
- cross entropy + dice

Recommended V1:

```text
L_final = CE(Y_final, Y_gt) + 0.5 * Dice(Y_final, Y_gt)
```

### `L_coarse`

Auxiliary segmentation loss on coarse logits:

- helps the initial head become usable before refinement

### `L_cons`

Cross-view consistency loss.

V1 practical option:

- use geometry-pruned correspondences
- encourage matched tokens across views to predict similar class distributions

One simple form:

```text
L_cons = KL(softmax(L_i^p) || stopgrad(softmax(M_i^p)))
```

or:

```text
L_cons = || softmax(L_i^p) - softmax(M_i^p) ||_1
```

Recommended V1 weights:

- `lambda_coarse = 0.3`
- `lambda_cons = 0.1`

## Training Plan

### Stage A: coarse head warmup

Train:

- `CoarseHead`

Freeze:

- `VGGT`

Setup:

- `4` views
- `224 x 224`
- no refinement yet
- no consistency yet

Goal:

- get stable single-view coarse part logits

### Stage B: alignment and refinement

Train:

- `CoarseHead`
- `SparseAlign`
- `RefineHead`

Freeze:

- `VGGT`

Setup:

- add geometry pruning
- add cross-view propagation
- add consistency loss

Goal:

- improve final accuracy
- improve cross-view consistency

## Hardware-Aware Settings

These settings fit the two-3090 setup and should be treated as part of the method design, not just implementation tricks.

### Recommended defaults

- image size: `224`
- views: `4`
- token grid: `16 x 16`
- `topk = 8`
- per-GPU batch: `1`
- grad accumulation: `4`
- mixed precision: `fp16` or `bf16`
- DDP: yes

### Strong recommendation: cache VGGT outputs offline

Since VGGT is frozen, precompute and store:

- token grid
- point map
- point confidence
- optionally camera or pose information

This gives:

- lower training memory
- faster iteration
- simpler debugging

Suggested cache format per sample:

```python
{
    "images": [N, 3, 224, 224],
    "token_grid": [N, 16, 16, 1024],
    "point_grid": [N, 16, 16, 3],
    "point_conf_grid": [N, 16, 16],
    "mask": [N, 224, 224],
    "meta": ...
}
```

## Suggested File Layout

```text
VG-AlignSeg/
├── configs/
│   ├── v1_train.yaml
│   └── v1_model.yaml
├── data/
│   ├── datasets/
│   └── cache_vggt_features.py
├── models/
│   ├── frozen_vggt.py
│   ├── coarse_head.py
│   ├── geometry_pruner.py
│   ├── sparse_align.py
│   ├── propagator.py
│   ├── refine_head.py
│   └── vg_alignseg.py
├── trainers/
│   └── train_v1.py
└── VG_ALIGNSEG_V1_DESIGN.md
```

## Suggested Ablations

Keep V1 ablations focused:

1. single-view coarse head only
2. geometry nearest-neighbor transfer without learnable alignment
3. sparse alignment without refinement
4. sparse alignment + refinement
5. no geometry pruning
6. logits propagation vs feature propagation

## Implementation Order

Build in this order:

1. frozen VGGT wrapper
2. token extraction and geometry downsampling
3. coarse head
4. training loop for single-view warmup
5. geometry pruner
6. sparse alignment
7. logits propagator
8. refine head
9. full stage-B training

## First Experiment Table

Use this exact starting point.

| Item | Value |
|---|---|
| Dataset | one category first |
| Views | 4 |
| Resolution | 224 |
| Backbone | frozen VGGT |
| Alignment space | 16 x 16 tokens |
| top-k | 8 |
| Propagation | logits only |
| Refine head | 2 residual blocks |
| Batch per GPU | 1 |
| Accumulation | 4 |
| Optimizer | AdamW |
| LR | 1e-3 |
| Weight decay | 1e-4 |

## Paper-Friendly Positioning

A clean summary for writing is:

> To make multi-view part segmentation practical on commodity GPUs, we freeze the geometry backbone, perform correspondence estimation only on low-resolution tokens, and restrict cross-view matching to geometry-pruned top-k candidates. This yields a lightweight yet geometry-aware alignment module that improves cross-view consistency without the cost of dense all-pairs matching.

## Next Step

The next concrete implementation step should be:

1. build `FrozenVGGTBackbone`
2. build `CoarseHead`
3. write a tiny forward-only sanity script on `4 x 224 x 224` inputs
4. confirm all tensor shapes above match in code
