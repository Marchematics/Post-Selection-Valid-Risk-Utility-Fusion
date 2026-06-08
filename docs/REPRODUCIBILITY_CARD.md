# Reproducibility Card

Paper: `Review-Budget-Constrained Calibration for Low-IoU UAV Object-Presence Triage`

## Environment

- Python 3
- `ultralytics` 8.4.53
- `torch` 2.8.0+cu128
- `numpy` 2.2.6
- `pandas` 2.3.2
- `scipy` 1.16.3
- `opencv-python` / `cv2` 4.12.0
- LaTeX: `latexmk` with `pdflatex`

## Main Protocol

- Main evidence builder: `scripts/l1_clear_accept_evidence.py`
- YOLO-family AITOD cache builder: `scripts/build_aitod_yolo_family_cache.py`
- Fixed-row evaluator: `scripts/fixed_cluster_candidate_eval.py`
- Contract-search helper: `scripts/cluster_contract_search_fast.py`
- Review-burden diagnostics: `scripts/review_burden_simulation.py` and `scripts/topk_review_simulation.py`
- AP diagnostics: `scripts/general_detection_ap_diagnostics.py`
- Figure generation: `scripts/plot_risk_utility_diagnostics.py`
- Metadata repair audit: `scripts/fix_public_metadata.py`

Main settings:

- Matching: ignored objects removed; class-aware greedy one-to-one matching
- IoU: `0.25`
- Main loss: `0.8 * miss_rate + 0.2 * min(FP/image, 300) / 300`
- Main review budget: `B = 25` FP/image
- Audit confidence: one-sided Hoeffding bound with confidence `0.95`
- Main audit target: `alpha = 0.16`
- RT-DETR selection target: `tilde alpha = 0.14`
- YOLO-family selection target: `tilde alpha = 0.13`
- Main RT-DETR row: AITOD `nms040_cap300`, threshold `0.125`
- Main YOLO-family row: AITOD `nms040_cap300`, threshold `0.015`

The calibration split is AITOD train. The audit split reported in the paper is AITOD validation.

## Cache Provenance

The audit consumes post-processed detector caches. Upstream detector inference, NMS prefiltering, resize/padding, and max-detection settings are fixed by those caches rather than re-estimated by the audit code. Raw public benchmark images and large local detector caches are not redistributed in the review package.

| Cache | File | Size | SHA-256 prefix |
| --- | --- | ---: | --- |
| AITOD RT-DETR val 640+960 cache | `pred_rows.parquet` | 20,914,264 | `ce85e2fbeda1` |
| AITOD RT-DETR val 640+960 cache | `gt_rows.parquet` | 223,283 | `f491287886e7` |
| AITOD RT-DETR val 640+960 cache | `image_meta.csv` | 86,046 | `7e5f1b0cb84` |
| AITOD RT-DETR train 640+960 cache | `pred_rows.parquet` | 80,835,199 | `d7551661aefd` |
| AITOD RT-DETR train 640+960 cache | `gt_rows.parquet` | 771,932 | `98225e888e5c` |
| AITOD RT-DETR train 640+960 cache | `image_meta.csv` | 347,822 | `627002941270` |
| AITOD YOLO-family val 640+960 cache | `pred_rows.parquet` | 3,336,603 | `d9276378250b` |
| AITOD YOLO-family train 640+960 cache | `pred_rows.parquet` | 12,716,913 | `2bdb4372d77d` |
| UAVDT RT-DETR-L/960 upgrade | `pred_rows.parquet` | 3,723,128 | `afd42134b781` |
| VisDrone RT-DETR-L/960 upgrade | `pred_rows.parquet` | 6,356,676 | `8ba69196776e` |

The AITOD YOLO-family caches were generated from the local AITOD-trained YOLO11n-family weights:

`/root/zjh_UAV_detection/experiments/aitod/runs/aitod_d10_baseline2stage_s2/weights/best.pt`

The paper and public repository release derived CSV summaries, scripts, and provenance, not the large detector-cache parquet files.

## Reported Outputs

- `output/tables/grsl_budget_constrained_main_summary.csv`
- `output/tables/l1_aitod_yolo11n_fixed_aitod_nms040_cap300_t0.125_image_operational_iou0.25_sequence_summary.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_t013_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_t014_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`
- `output/tables/review_burden_simulation_v2.csv`
- `output/tables/topk_review_simulation.csv`
- `output/tables/ap_diagnostics_aitod_val_positive_summary.csv`
- `output/tables/localization_tier_boundary_summary.csv`
- `output/tables/cluster_size_summary.csv`
- `output/tables/metadata_corrections_audit.csv`

## Known Naming Issue

Some historical filenames retain `aitod_uavdt` as a legacy contract/script tag from earlier UAVDT-oriented routing. The `dataset` field inside the released tables is authoritative and has been corrected to `aitod` where appropriate; corrections are listed in `output/tables/metadata_corrections_audit.csv`.
