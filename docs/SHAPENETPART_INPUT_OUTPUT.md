# ShapeNetPart Input / Output

## What one training sample looks like

For `VG-AlignSeg V1`, one sample should be a set of views from the same object.

Recommended first setting:

- object category: `Chair`
- number of views: `4`
- image size: `224 x 224`
- part labels: one mask per view

The task IO is:

```text
Input:
    4 RGB views of the same object
    shape = [B, 4, 3, 224, 224]

Supervision:
    4 part masks
    shape = [B, 4, 224, 224]

Model outputs:
    coarse per-view logits
    shape = [B, 4, K, 224, 224]

    refined per-view logits
    shape = [B, 4, K, 224, 224]
```

where `K` is the number of part classes for the chosen category.

## Internals

Frozen VGGT produces:

- token grid: `[B, 4, 16, 16, 2048]`
- point grid: `[B, 4, 16, 16, 3]`
- geometry confidence: `[B, 4, 16, 16]`

Then:

1. `CoarseHead` predicts low-resolution part logits.
2. `GeometryPruner` finds sparse candidates across views.
3. `SparseAlign` computes top-k alignment weights.
4. `LogitPropagator` transfers logits from source views to target view.
5. `RefineHead` outputs the final part mask for each view.

## Visualization

See:

- [shapenetpart_v1_io_demo.png](/home/lyx/curriculum/computer_vision/VG-AlignSeg/docs/assets/shapenetpart_v1_io_demo.png)

This figure is a local task illustration, not a real ShapeNetPart render, because the dataset is not yet downloaded in the workspace.
