# Dataset Source And Split

This document records the dataset origin and the exact train/test split used by
the final VG-AlignSeg V4 results.

## Source

The eight-view data used in this repository comes from the Hugging Face dataset
`luyu1021/vg-alignseg`:

- Dataset page: `https://huggingface.co/datasets/luyu1021/vg-alignseg`
- Downloaded archive: `result.zip`
- Local unpacked root: `data/vg-alignseg-dataset`

The category metadata is copied from the same dataset repository:

- `final_results/metadata/category_models_list.txt`
- `final_results/metadata/object_category_map.json`
- `final_results/metadata/category_summary.json`

The metadata file lists 46 object categories and 2339 object ids. The final
loader keeps 2187 valid eight-view objects after checking that each object has
the expected color views and part-mask folders.

## Local Object Format

Each object is stored as:

```text
data/vg-alignseg-dataset/<object_id>_views/
  color/
  part_mask/
    actor_<id>/
  point_cloud/
  label1/
```

The V4 experiments use these eight color views in this fixed order:

```text
view_1_right_front_top.png
view_2_right_front_bottom.png
view_3_right_back_top.png
view_4_right_back_bottom.png
view_5_left_front_top.png
view_6_left_front_bottom.png
view_7_left_back_top.png
view_8_left_back_bottom.png
```

The dataset loader is `data.result_multiview.ResultMultiViewDataset`. An object
is considered valid if:

- its directory name ends with `_views`;
- it contains `color/` and `part_mask/`;
- all eight expected color-view files exist;
- at least one `part_mask/actor_*` directory exists.

## Final Split

The final reported V4 result uses a continuous, non-shuffled split:

| Split | Count |
| --- | ---: |
| train | 2000 |
| val | 0 |
| test | 187 |
| total valid | 2187 |

Split arguments:

```text
train_count=2000
val_count=0
test_count=-1
shuffle_split=false
```

Important: this is not a random split. Objects are sorted lexicographically by
their `<object_id>_views` directory name, then the first 2000 objects are used
for training and the remaining 187 objects are used for testing. The recorded
`split_seed=42` in some manifests has no effect unless `--shuffle-split` is
enabled.

The exact manifests are stored in git:

- Train manifest:
  `final_results/training/v4_top2_phase2_random_copy/train_manifest.json`
- Test manifest:
  `final_results/evaluations/v4_target_only_top4_best4500/test_manifest.json`
- Machine-readable split summary:
  `final_results/split_summary.json`

## Object Id Ranges

Because sorting is lexicographic rather than numeric, the id ranges look a
little unintuitive:

| Split | First object id | Last object id |
| --- | ---: | ---: |
| train | 100013 | 47747 |
| test | 47808 | 9996 |

First 10 test ids:

```text
47808, 47817, 47853, 47926, 47944, 47954, 47963, 47976, 48010, 48013
```

Last 10 test ids:

```text
9388, 9393, 9410, 960, 9748, 9912, 991, 9960, 9992, 9996
```

## Test Category Distribution

The final test split covers 12 object categories:

| Category | Test objects |
| --- | ---: |
| Bottle | 16 |
| Box | 1 |
| Chair | 2 |
| Clock | 31 |
| Display | 7 |
| Door | 35 |
| Faucet | 14 |
| Keyboard | 1 |
| Laptop | 5 |
| Microwave | 12 |
| Oven | 7 |
| StorageFurniture | 56 |

## Regenerate Split Summary

```bash
python scripts/summarize_dataset_split.py \
  --data-root data/vg-alignseg-dataset \
  --category-map final_results/metadata/object_category_map.json \
  --output final_results/split_summary.json \
  --views 8 \
  --train-count 2000 \
  --val-count 0 \
  --test-count -1
```

To create a random split for a future experiment, add:

```bash
--shuffle-split --split-seed 42
```

That random split is not the split used by the final reported V4 metrics.
