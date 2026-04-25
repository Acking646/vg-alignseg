# ShapeNetPart Real Data

## Recommended local path

```text
/home/lyx/datasets/ShapeNetPart/shapenetcore_partanno_segmentation_benchmark_v0_normal
```

## Download

```bash
cd /home/lyx/curriculum/computer_vision/VG-AlignSeg
python scripts/download_shapenetpart.py
```

The script tries:

1. the original Stanford archive URL
2. a fallback research mirror

and verifies the archive SHA256 unless `--skip-sha256` is passed.

## Real sample visualization

```bash
cd /home/lyx/curriculum/computer_vision/VG-AlignSeg
python scripts/visualize_shapenetpart_real_sample.py --category Chair --split train --index 0
```

Outputs:

- a rendered PNG under `docs/assets/`
- an `.npz` file containing:
  - `images`: `[4, 224, 224, 3]`
  - `masks`: `[4, 224, 224]`
  - `view_names`

## Input / target definition

For the first VG-AlignSeg prototype with ShapeNetPart:

- input: `4` rendered views of the same object
- target: `4` per-view 2D part masks

This lets us build a clean synthetic benchmark before moving to a more complex source such as PartNet.
