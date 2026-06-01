# Post-Selection-Valid Risk-Utility Fusion for UAV Object Detection

This repository contains the reproducibility materials for the Letter "Post-Selection-Valid Risk-Utility Fusion for UAV Object Detection".

The repository is anonymized at the file/path level. It contains only relative paths and does not include local workstation paths, user names, logs, manuscript submission files, or raw UAV image files.

## Contents

- `scripts/`: Python code for finite-sample miss-risk certification, finite-family post-selection audits, and supporting risk-utility experiments.
- `data/caches/`: derived detector-cache tables for UAVDT and VisDrone validation images. Each cache contains `gt_rows.parquet`, `pred_rows.parquet`, and `image_meta.csv`.
- `data/manifests/`: deterministic split manifests for random-half, image-hash, and sequence-hash checks.
- `output/tables/`: CSV result tables used to produce the main manuscript tables and Fig. 1.
- `paper/figures/`: Fig. 1 source script and generated PDF/PNG.
- `docs/`: source-data map and data-availability notes.

## Data Scope

The raw UAVDT and VisDrone images are not redistributed. The included cache tables are derived tabular data:

- `gt_rows.parquet`: public benchmark boxes converted to a common schema.
- `pred_rows.parquet`: frozen detector predictions at the cached resolutions.
- `image_meta.csv`: image-level counts and small-object metadata.

These tables are sufficient to rerun the finite-sample audits reported in the Letter.

## Environment

Python 3.10 or later is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Required Python packages are `numpy`, `pandas`, `scipy`, `matplotlib`, and `pyarrow`.

## Reproducing the Main Tables

From the repository root:

```bash
bash run_reproduce_tables.sh
```

The script regenerates the main UAVDT finite-family rows, the risk-margin variant, the family-corrected certificate, the sequence-support stress row, VisDrone relaxed-target rows, and IoU 0.50 boundary rows. Outputs are written to `output/tables/`.

The included CSV files are the exact tables used for the submitted manuscript.

For a quick smoke check:

```bash
python scripts/post_selection_family_audit.py \
  --dataset uavdt \
  --splits image_lockbox \
  --alpha 0.16 \
  --iou 0.25 \
  --out-prefix smoke_check
```

## Cluster-Ratio Main Audit

The current manuscript title is "Cluster-Robust Post-Selection Risk-Utility Fusion for UAV Object Detection". The cluster-ratio audit adds image-unit calibration for object-weighted miss risk using the statistic `M_g - alpha N_g` and fixed-sequence LTT threshold selection.

Reproduce the main cluster-ratio row with:

```bash
bash run_reproduce_cluster_ratio.sh
```

Expected selected row: UAVDT image-cluster certificate, `support_floor`, threshold `0.03125`, `z_upper=-0.0073`, `fp_upper=52.1`, lockbox risk `0.0306`, precision `0.3741`, and `36.9` FP/image.
