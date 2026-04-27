# VG-AlignSeg: Source-Guided Multi-View Part Transfer with Frozen Geometric Tokens

## Abstract

We study eight-view part segmentation for object-centric multi-view data. A
direct global actor-id classifier performs poorly because actor ids are local to
each object rather than globally meaningful semantic categories. We therefore
reformulate multi-view segmentation as source-guided part transfer: given a
source-view binary mask for one part, the model predicts the same part in the
remaining views. Our system freezes a VGGT encoder and trains a lightweight
decoder that combines source-mask prompts, VGGT token prototypes, point-map
geometry, RGB high-resolution features, and iterative refinement. On a random
2000/187 train/test split, the strict target-only top4 protocol reaches 46.0
actor mIoU and 99.37 pixel accuracy. An oracle full-view protocol that copies
source-view ground truth reaches 95.8 mIoU, showing that multi-source selection
and composition are coherent while also revealing that target-only transfer
remains the central challenge.

## 1. Introduction

Multi-view part segmentation needs two forms of consistency: pixel-accurate
boundaries within each view and actor correspondence across views. In the
VG-AlignSeg data, the same object appears in eight views and each visible part is
annotated by an actor id. A natural baseline is to train a semantic segmentation
network over all actor ids. This is weak because actor ids are instance-local:
`actor_3` is not necessarily the same semantic part for every object. The
classifier therefore learns a mixed global label space and cannot reliably
generalize to unseen objects.

We adopt a different formulation. Instead of predicting every actor id directly,
the model receives a source mask for one actor and transfers that mask to the
other views. The same binary transfer head can be applied to all actors of an
object and the final segmentation is obtained by composing actor logits. This
makes the task closer to cross-view object/part correspondence and removes the
need for global actor-id semantics.

## 2. Related Work

VGGT provides frozen multi-view geometry-aware representations, including token
features and point-map predictions. VGGT-Segmentor further shows that a source
mask can be injected into VGGT features, combined with point-guided prediction,
and refined by a segmentation decoder for cross-view segmentation. Our method is
inspired by this direction, but differs in problem setting and output structure:
we operate on eight object-centric views, predict arbitrary local part actors,
aggregate multiple source views, and evaluate target-only transfer without
counting copied source masks.

## 3. Method

Given eight images \(I_1,\ldots,I_8\), a part actor \(a\), and one or more source
views \(S_a\), the model predicts binary masks \(\hat{M}_{a,v}\) for all views
v. The full multi-part prediction is built by repeating this process for every
actor and assigning each pixel to the actor with the strongest foreground logit.

Architecture:

- Frozen VGGT backbone: extracts multi-view token grids, point grids, and
  confidence maps.
- Source prompt encoder: embeds the binary source mask and source-view indicator.
- Prototype branch: computes foreground/background token prototypes from the
  source mask and scores every target token by similarity.
- Point branch: uses VGGT point maps to provide geometry-aware coarse transfer.
- High-resolution decoder: fuses low-resolution VGGT cues with RGB skip features
  in a U-Net/DeepLab-style refinement head.
- Boundary auxiliary head: encourages sharper mask edges.
- Iterative refinement: feeds the previous binary logit back into the decoder.

Training uses selected top-k source views per actor. In the final run we use
top4 source prompts, train each source prompt independently for memory stability,
and exclude prompted source views from the transfer loss.

## 4. Experiments

Dataset and split:

- Dataset: `data/vg-alignseg-dataset`
- Views: 8
- Resolution: 224
- Random split seed: 42
- Train/test: 2000/187 objects

Final training:

- Frozen VGGT encoder
- Hidden dimension: 128
- Refinement iterations: 2
- Optimizer: AdamW
- Learning rate: 8e-5
- Losses: BCE, focal, Dice, Tversky, boundary, boundary auxiliary
- Checkpoint: step 3000 from `v4_random2000_top4_independent_seed42`

Metrics:

- Strict target-only actor mIoU: actor-wise IoU on non-source views only.
- Pixel accuracy: pixel agreement on evaluated target pixels.
- Oracle full-view mIoU: full 8-view metric with source GT copied into source
  views. This is an upper bound, not the main transfer score.

## 5. Results

| Protocol | Source GT in metric | Split | Objects | mIoU | Pixel Acc |
| --- | --- | --- | ---: | ---: | ---: |
| Top4 strict target-only | No | random seed 42 | 187 | 0.4602 | 0.9937 |
| Top4 oracle full-view | Yes | earlier fixedviz eval | 187 | 0.9578 | 0.9984 |
| Top3 oracle full-view | Yes | earlier fixedviz eval | 187 | 0.8977 | 0.9969 |
| Top2 full-view copy run | Yes | earlier random copy run | 187 | 0.8315 | 0.9948 |

The strict result is the honest no-GT-transfer number. The oracle result explains
why the visualizations look strong when the source views are included: top-k
source selection and logit composition work, but much of the metric is helped by
known source masks and easier visible regions.

## 6. Discussion

The main lesson is that source-guided transfer is the right formulation for this
dataset, but the current decoder still struggles with target-only unseen views.
The gap between 95.8 oracle mIoU and 46.0 strict mIoU suggests three limitations:

- Source-view selection uses GT visibility, so deployment needs a learned source
  proposal or a user-provided prompt.
- Binary transfer is trained per actor, but final composition can still confuse
  adjacent small parts.
- The model is limited to 224 resolution and a lightweight decoder, which hurts
  fine boundaries and tiny components.

## 7. Contributions

1. We identify global actor-id segmentation as a poor formulation for local
   actor labels.
2. We introduce a source-guided binary part transfer formulation for eight-view
   object-centric segmentation.
3. We implement a frozen-VGGT transfer head with mask prompts, token prototypes,
   point-map geometry, high-resolution refinement, boundary supervision, and
   top-k multi-source aggregation.
4. We define a strict target-only evaluation protocol that avoids mixing source
   GT into the metric.
5. We provide an oracle/full-view upper-bound analysis to separate composition
   quality from true cross-view transfer quality.

## 8. What To Write In A Submission

The strongest honest paper angle is not "we already beat SOTA"; the current
strict result does not support that. The publishable angle is:

- Problem insight: local actor ids break standard semantic segmentation.
- Method novelty: source-guided part transfer over eight VGGT-aligned views.
- Evaluation novelty: target-only protocol that prevents source-GT leakage.
- Empirical finding: huge oracle/strict gap, exposing the real bottleneck in
  multi-view part transfer.

For a stronger submission, the next experiment should improve strict transfer:
train at higher resolution, add a learned source selector, add cross-actor
competition during training, and report category-level part grouping if semantic
part names can be recovered.

