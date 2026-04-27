# VG-AlignSeg: Source-Guided Multi-View Part Transfer with Frozen Geometric Tokens

## Abstract

We study eight-view part segmentation for object-centric multi-view data. A
direct global actor-id classifier performs poorly because actor ids are local to
each object rather than globally meaningful semantic categories. We therefore
reformulate multi-view segmentation as source-guided part transfer: given one or
more source-view binary masks for a part, the model predicts the same part across
all views and composes the final multi-part segmentation. Our system freezes a
VGGT encoder and trains a lightweight decoder that combines source-mask prompts,
VGGT token prototypes, point-map geometry, RGB high-resolution features, and
iterative refinement. On the final staged/oracle top4 protocol, VG-AlignSeg
reaches 95.78 part mIoU, 92.42 category-balanced IoU, 95.88
granularity-balanced IoU, and 99.84 cross-view consistency accuracy. A stricter
no-source-copy diagnostic reaches 46.02 target-only actor mIoU, showing that the
staged protocol is strong for guided multi-view segmentation while fully
GT-free target transfer remains the central challenge.

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
the model receives source masks for one actor and transfers that mask to the
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
aggregate multiple source views, and explicitly distinguish staged/oracle
evaluation from strict target-only transfer.

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

## 4. Progressive Training

The final result was obtained through a progressive process:

1. Direct multi-class actor segmentation validated the data pipeline and overfit
   capacity, but generalized poorly because actor ids are local.
2. V4 source-guided transfer converted the problem to binary part transfer from
   source masks.
3. Multi-source aggregation improved robustness by using top-k visible source
   views per actor.
4. The final top4 staged/oracle evaluation copies source-view masks into source
   views and evaluates the full eight-view composed segmentation.

Final staged checkpoint:

- `outputs/v4_result_ddp2_multisource_top2_phase2_random_copy/best.pt`
- Evaluation script: `scripts/evaluate_v4_staged_oracle.py`

## 5. Experiments

Dataset and split:

- Dataset: `data/vg-alignseg-dataset`
- Categories: recovered from `luyu1021/vg-alignseg/category_models_list.txt`
- Views: 8
- Resolution: 224
- Train/test: 2000/187 objects

Metrics:

- `iou-object-category`: macro average over object categories.
- `iou-granularity`: macro average over actor-count buckets:
  `coarse_1_2_parts`, `medium_3_4_parts`, and `fine_5plus_parts`.
- `iou-part`: full-view segmentation mIoU.
- `cross-view consistency acc`: full eight-view pixel accuracy.

## 6. Results

| Protocol | iou-object-category | iou-granularity | iou-part | cross-view consistency acc |
| --- | ---: | ---: | ---: | ---: |
| Top4 staged/oracle | 0.9242 | 0.9588 | 0.9578 | 0.9984 |
| Top4 strict target-only diagnostic | N/A | N/A | 0.4602 | 0.9937 |

The staged/oracle result explains why the fixedviz visualizations look strong:
top-k source selection, source-mask prompting, and logit composition work very
well when source masks are provided and source views are included in the metric.
The strict diagnostic is lower because it removes source views from the score and
measures only target transfer.

## 7. Discussion

The strongest current paper angle is guided multi-view part segmentation. The
method gives a solid, high-performing staged/oracle system and a transparent
diagnostic protocol. The gap between 95.78 staged mIoU and 46.02 strict mIoU is
important: it shows that the architecture can compose high-quality guided
multi-view masks, while fully automatic source-free target transfer remains open.

Limitations:

- Source-view selection uses GT visibility in the staged/oracle protocol.
- The final metric includes copied source-view masks.
- Strict target-only transfer remains substantially weaker.
- Actor ids are local and not guaranteed to be semantic part names.

## 8. Contributions

1. We identify global actor-id segmentation as a poor formulation for local
   actor labels.
2. We introduce a source-guided binary part transfer formulation for eight-view
   object-centric segmentation.
3. We implement a frozen-VGGT transfer head with mask prompts, token prototypes,
   point-map geometry, high-resolution refinement, boundary supervision, and
   top-k multi-source aggregation.
4. We provide category-balanced and granularity-balanced staged/oracle metrics.
5. We retain a strict target-only diagnostic to expose the real transfer
   bottleneck.

## 9. Submission Framing

Do not frame the result as a fully GT-free SOTA segmentation system. The honest
and stronger framing is: VG-AlignSeg is a guided multi-view part segmentation
system with strong top4 source-guided performance, explicit category/granularity
analysis, and a transparent target-only diagnostic.

