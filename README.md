# Post-Selection-Valid Risk-Utility Fusion for UAV Object Detection

This repository contains the reproducibility materials for the Letter "Post-Selection-Valid Risk-Utility Fusion for UAV Object Detection".

The repository is anonymized at the file/path level. It contains only relative paths and does not include local workstation paths, user names, manuscript submission files, or raw UAV image files.

## Contents

- `scripts/`: Python code for finite-sample miss-risk certification, finite-family post-selection audits, cluster-ratio diagnostics, and supporting risk-utility experiments.
- `data/caches/`: derived detector-cache tables for UAVDT and VisDrone. Each cache contains `gt_rows.parquet`, `pred_rows.parquet`, and `image_meta.csv`.
- `data/manifests/`: deterministic split manifests for random-half, image-hash, sequence-hash, validation-only, and train+val development checks.
- `output/tables/`: CSV result tables used to produce the manuscript tables and Fig. 1.
- `paper/figures/`: Fig. 1 source script and generated PDF/PNG.
- `docs/`: source-data map and data-availability notes.

## Data Scope

The raw UAVDT and VisDrone images are not redistributed. The included cache tables are derived tabular data:

- `gt_rows.parquet`: public benchmark boxes converted to a common schema.
- `pred_rows.parquet`: frozen detector predictions at cached resolutions.
- `image_meta.csv`: image-level counts and small-object metadata.

These tables are sufficient to rerun the finite-sample audits reported in the Letter.

## Main Validation-Only Audit

The main uncontaminated result is the UAVDT validation-only finite-family audit. It compares raw RT-DETR-L/960 with the selected source-support family row on the same image-hash split.

From the repository root:

```bash
bash run_reproduce_tables.sh
```

Expected main validation-only row: `support_floor`, CP-U `0.1598`, FP-U `140.2`, evaluation risk `0.1338`, precision `0.2508`, and `83.0` FP/image. Raw 960 on the same split has evaluation risk `0.1402`, precision `0.1369`, and `174.0` FP/image.

## Cluster-Ratio Diagnostics

The cluster-ratio diagnostic uses image units and the statistic `M_g - alpha N_g` to avoid object-iid assumptions. It is reported as a diagnostic rather than the headline.

```bash
bash run_reproduce_cluster_ratio.sh
```

Expected validation-only cluster-ratio outcome: abstention (`z_upper=15.558`, FP-U `335.1`). Expected train+val development row: `support_floor`, threshold `0.03125`, `z_upper=-0.0073`, FP-U `52.1`, evaluation risk `0.0306`, precision `0.3741`, and `36.9` FP/image. The train+val row is marked development stress because it mixes source splits and is not sequence-disjoint.

## Environment

Python 3.10 or later is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Required Python packages are `numpy`, `pandas`, `scipy`, `matplotlib`, and `pyarrow`.
