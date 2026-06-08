# Reproducibility Card

Paper: `Cluster-Aware Risk Audit for Low-IoU UAV Object-Presence Triage`

## Environment

- Python 3
- `ultralytics` 8.4.53
- `torch` 2.8.0+cu128
- `numpy` 2.2.6
- `pandas` 2.3.2
- `scipy` 1.16.3
- `opencv-python` / `cv2` 4.12.0

## Main Audit Protocol

- Fixed-row evaluator: `scripts/fixed_cluster_candidate_eval.py`
- Cluster family replay: `scripts/aitod_multiunit_family_replay_fast.py`
- Cluster-search helper: `scripts/cluster_contract_search_fast.py`
- AP diagnostics: `scripts/general_detection_ap_diagnostics.py`
- Review burden diagnostics: `scripts/review_burden_simulation.py` and `scripts/topk_review_simulation.py`
- Metadata repair audit: `scripts/fix_public_metadata.py`
- Matching: ignored objects removed; class-aware greedy one-to-one matching
- Main fixed row: AITOD `nms040_cap300`, threshold `0.125`, IoU `0.25`
- Main loss: `0.8 * miss_rate + 0.2 * min(FP/image, 300) / 300`
- Audit confidence: one-sided Hoeffding bound with confidence `0.95`
- Main target: `alpha = 0.16`
- Train-side replay: 3 NMS-family contracts x 30 thresholds x 3 unit definitions, Bonferroni-corrected Hoeffding bounds, selection target `0.14`

## Cache Provenance

The audit consumes post-processed detector caches. Upstream NMS, score prefiltering, resize/padding, and max-detection settings are fixed by those caches rather than re-estimated by the audit code.

| Cache | File | Size | SHA-256 prefix |
| --- | --- | ---: | --- |
| AITOD val 640+960 cache | `pred_rows.parquet` | 20,914,264 | `ce85e2fbeda1` |
| AITOD val 640+960 cache | `gt_rows.parquet` | 223,283 | `f491287886e7` |
| AITOD val 640+960 cache | `image_meta.csv` | 86,046 | `7e5f1b0cb84` |
| AITOD train 640+960 cache | `pred_rows.parquet` | 80,835,199 | `d7551661aefd` |
| AITOD train 640+960 cache | `gt_rows.parquet` | 771,932 | `98225e888e5c` |
| AITOD train 640+960 cache | `image_meta.csv` | 347,822 | `627002941270` |
| UAVDT RT-DETR-L/960 upgrade | `pred_rows.parquet` | 3,723,128 | `afd42134b781` |
| UAVDT RT-DETR-L/960 upgrade | `gt_rows.parquet` | 189,835 | `b21dd13a7cfe` |
| UAVDT YOLO11m/640 default | `pred_rows.parquet` | 2,232,526 | `b661c071bbc4` |
| UAVDT combined 640+960 | `pred_rows.parquet` | 5,901,761 | `5e3c186cc7dc` |
| VisDrone RT-DETR-L/960 upgrade | `pred_rows.parquet` | 6,356,676 | `8ba69196776e` |
| UAVDT YOLOv8L/640 baseline | `pred_rows.parquet` | 7,263,516 | `02c84a606efb` |

The released package includes derived CSV/Parquet summaries used by the paper.
Raw public benchmark images and large local detector caches are not redistributed.
The repository README states that rerunning all raw detector inference requires
the public datasets plus compatible local detector-cache generation.
Some historical filenames retain `aitod_uavdt` as a legacy contract/script tag;
the `dataset` field inside each released table is authoritative.

## Reported Outputs

- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/aitod_multiunit_family_replay_nms3_fast_selected.csv`
- `output/tables/review_burden_simulation_v2.csv`
- `output/tables/topk_review_simulation.csv`
- `output/tables/ap_diagnostics_aitod_val_positive_summary.csv`
- `output/tables/localization_tier_boundary_summary.csv`
- `output/tables/cluster_size_summary.csv`
- `output/tables/aitod_loss_cap_alpha_sensitivity.csv`
- `output/tables/metadata_corrections_audit.csv`
