# VG-AlignSeg: Source-Guided Multi-View Part Transfer with Frozen Geometric Tokens

## Abstract

We study eight-view part segmentation for object-centric multi-view data. A
direct global actor-id classifier performs poorly because actor ids are local to
each object rather than globally meaningful semantic categories. We therefore
reformulate multi-view segmentation as source-guided part transfer: given one or
more source-view binary masks for a part, the model predicts the same part on
target views and composes the final multi-part segmentation. Our system freezes
a VGGT encoder and trains a lightweight decoder that combines source-mask
prompts, VGGT token prototypes, point-map geometry, RGB high-resolution
features, and iterative refinement. Under a strict top4 target-only evaluation,
VG-AlignSeg reaches 61.71 part mIoU, 56.14 category-balanced IoU, 53.44
granularity-balanced IoU, and 99.75 target pixel accuracy. A source-copy
full-view reference reaches 95.78 part mIoU, showing that guided composition is
strong while source-free target transfer remains the central challenge.

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
the model receives source masks for one actor and transfers that mask to other
views. The same binary transfer head can be applied to all actors of an object,
and the final segmentation is obtained by composing actor logits. This makes the
task closer to cross-view object/part correspondence and removes the need for
global actor-id semantics.

## 2. Related Work

VGGT provides frozen multi-view geometry-aware representations, including token
features and point-map predictions. VGGT-Segmentor further shows that a source
mask can be injected into VGGT features, combined with point-guided prediction,
and refined by a segmentation decoder for cross-view segmentation. Our method is
inspired by this direction, but differs in problem setting and output structure:
we operate on eight object-centric views, predict arbitrary local part actors,
aggregate multiple source views, and explicitly evaluate non-source target
views.

## 3. Method

Given eight images \(I_1,\ldots,I_8\), a part actor \(a\), and source views
\(S_a\), the model predicts binary masks \(\hat{M}_{a,v}\) for target views v.
The full multi-part prediction is built by repeating this process for every
actor and assigning each pixel to the actor with the strongest foreground logit.

Architecture:

- Frozen VGGT backbone: extracts multi-view token grids, point grids, and
  confidence maps.
- Source prompt encoder: embeds the binary source mask and source-view
  indicator.
- Prototype branch: computes foreground/background token prototypes from the
  source mask and scores every target token by similarity.
- Point branch: uses VGGT point maps to provide geometry-aware coarse transfer.
- High-resolution decoder: fuses low-resolution VGGT cues with RGB skip
  features in a U-Net/DeepLab-style refinement head.
- Boundary auxiliary head: encourages sharper mask edges.
- Iterative refinement: feeds the previous binary logit back into the decoder.

## 4. Progressive Training

The final result was obtained through a progressive process:

1. Direct multi-class actor segmentation validated the data pipeline and
   overfit capacity, but generalized poorly because actor ids are local.
2. V4 source-guided transfer converted the problem to binary part transfer from
   source masks.
3. Multi-source aggregation improved robustness by using top-k visible source
   views per actor.
4. The final strict target-only evaluation uses top4 source masks as prompts and
   scores only non-source target views.

Final checkpoint:

- `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`
- Main evaluation script: `scripts/evaluate_v4_dynamic_target_only.py`
- Result summarizer: `scripts/summarize_target_only_results.py`

## 5. Experiments

Dataset and split:

- Dataset: `data/vg-alignseg-dataset`
- Categories: recovered from `luyu1021/vg-alignseg/category_models_list.txt`
- Views: 8
- Resolution: 224
- Valid objects: 2187
- Train/val/test: 2000/0/187 objects
- Split policy: lexicographic object-directory order, no shuffle. The first
  2000 valid objects are used for training and the remaining 187 objects are
  used for testing.

Metrics:

- `iou-object-category`: macro average over object categories.
- `iou-granularity`: macro average over actor-count buckets:
  `coarse_1_2_parts`, `medium_3_4_parts`, and `fine_5plus_parts`.
- `iou-part`: actor-wise IoU on non-source target views.
- `cross-view consistency acc`: target-view pixel accuracy.

## 6. Results

| Protocol | iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| --- | ---: | ---: | ---: | ---: |
| Top4 strict target-only | 0.5614 | 0.5344 | 0.6171 | 0.9975 |
| Top4 source-copy full-view reference | 0.9242 | 0.9588 | 0.9578 | 0.9984 |

The strict target-only result measures actual transfer to non-source views. It
is therefore the main result. The source-copy reference explains why the old
fixedviz images look stronger: it includes copied source masks in the full-view
metric and serves as an upper guided-composition reference.

Per-granularity strict target-only behavior:

| Granularity | Objects | mIoU |
| --- | ---: | ---: |
| coarse_1_2_parts | 94 | 0.7775 |
| medium_3_4_parts | 77 | 0.4771 |
| fine_5plus_parts | 16 | 0.3485 |

## 7. Discussion

The strongest current paper angle is guided multi-view part segmentation under
local actor labels. The method gives a solid target-only transfer system and a
transparent source-copy reference. The gap between 61.71 target-only mIoU and
95.78 source-copy full-view mIoU is important: it shows that the architecture can
compose high-quality guided masks when source views are available, while
occluded, small, and fine-grained target parts remain difficult.

Limitations:

- Source-view selection uses GT visibility.
- The system is source-guided and requires source masks as prompts.
- Fine-grained objects with five or more parts are much harder than coarse
  two-part objects.
- Actor ids are local and not guaranteed to be semantic part names.

## 8. Contributions

1. We identify global actor-id segmentation as a poor formulation for local
   actor labels.
2. We introduce a source-guided binary part transfer formulation for eight-view
   object-centric segmentation.
3. We implement a frozen-VGGT transfer head with mask prompts, token prototypes,
   point-map geometry, high-resolution refinement, boundary supervision, and
   top-k multi-source aggregation.
4. We report strict target-only, category-balanced, and granularity-balanced
   metrics.
5. We retain a source-copy full-view reference to separate target transfer
   quality from source-view reconstruction.

## 9. Submission Framing

Do not frame the result as a fully automatic source-free SOTA segmentation
system. The honest and stronger framing is: VG-AlignSeg is a guided multi-view
part transfer system with a strict target-only evaluation, explicit
category/granularity analysis, and a transparent source-copy reference.
