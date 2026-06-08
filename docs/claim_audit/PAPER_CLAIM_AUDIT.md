# Paper Claim Audit Report

**Date**: 2026-06-08  
**Auditor**: local executor audit against released CSV outputs  
**Paper**: `Review-Budget-Constrained Calibration for Low-IoU UAV Object-Presence Triage`  
**Overall verdict**: PASS for the quantitative claims checked below.

## Bibliography Invariant

`paper/references.bib` was not edited. Last checked timestamp and size:

`2026-06-07 20:59:24.037793361 +0800`, `7591` bytes.

## Main Budget-Constrained Calibration Claims

Primary source:

- `output/tables/grsl_budget_constrained_main_summary.csv`

Verified rounded values:

| Claim location | Paper value | Evidence status |
|---|---:|---|
| RT-DETR selected row | `nms040_cap300@0.125` | exact match |
| RT-DETR train target | `0.14` | exact match |
| RT-DETR train upper / FP-image | `0.1336` / `10.79` | exact/rounding OK |
| RT-DETR validation upper / FP-image | `0.1496` / `12.03` | exact/rounding OK |
| RT-DETR validation precision / miss | `0.4886` / `0.2072` | exact/rounding OK |
| YOLO-family selected row | `nms040_cap300@0.015` | exact match |
| YOLO-family train target | `0.13` | exact match |
| YOLO-family train upper / FP-image | `0.1273` / `20.05` | exact/rounding OK |
| YOLO-family validation upper / FP-image | `0.1541` / `20.77` | exact/rounding OK |
| YOLO-family validation precision / miss | `0.3511` / `0.2251` | exact/rounding OK |
| RT-DETR unconstrained diagnostic | `nms040@0.02`, train upper `0.1051`, FP-image `36.89`, precision `0.2447`, miss `0.1134` | exact/rounding OK |

Interpretation checked: the paper claims per-cache calibration, not detector-agnostic threshold transfer.

## Fixed-Threshold Transfer Boundary

Primary source:

- `output/tables/l1_aitod_yolo11n_fixed_aitod_nms040_cap300_t0.125_image_operational_iou0.25_sequence_summary.csv`

Verified rounded values:

| Claim location | Paper value | Evidence status |
|---|---:|---|
| Applying RT-DETR threshold to YOLO-family cache | `U_H=0.2435` | exact/rounding OK |
| YOLO-family fixed-threshold precision / FP-image | `0.6862` / `4.25` | exact/rounding OK |
| YOLO-family fixed-threshold miss | `0.3590` | exact/rounding OK |

Interpretation checked: the fixed threshold fails the loss audit despite low clutter; this supports detector-specific calibration rather than a universal threshold.

## Conservative-Target Boundary

Primary sources:

- `output/tables/aitod_multiunit_family_replay_nms3_fast_selected.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_t014_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`

Verified rounded values:

| Claim location | Paper value | Evidence status |
|---|---:|---|
| RT-DETR relaxed train target `0.16` selects | `nms040_cap300@0.175` | exact match |
| RT-DETR relaxed-row validation upper | `0.1696` | exact/rounding OK |
| RT-DETR relaxed-row train FP-image | `7.83` | exact/rounding OK |
| YOLO-family train target `0.14` validation upper | `0.1675` | exact/rounding OK |
| YOLO-family train target `0.16` validation upper | `0.1872` | exact/rounding OK |

Interpretation checked: stricter `\tilde{\alpha}` values are margin choices exposed by train-to-validation replay, not validation-best thresholds.

## Review Burden and AP Diagnostics

Primary sources:

- `output/tables/review_burden_simulation_v2.csv`
- `output/tables/topk_review_simulation.csv`
- `output/tables/ap_diagnostics_aitod_val_positive_summary.csv`
- `paper/figures/review_budget_tradeoff.pdf`
- `scripts/plot_risk_utility_diagnostics.py`

Verified rounded values:

| Claim location | Paper value | Evidence status |
|---|---:|---|
| Raw RT-DETR boxes / FP-image / miss / precision | `65.55` / `53.53` / `0.1710` / `0.1834` | exact/rounding OK |
| NMS+cap boxes / FP-image / miss / precision | `23.53` / `12.03` / `0.2072` / `0.4886` | exact/rounding OK |
| Review-box reduction | `64.1%` | exact/rounding OK |
| Top-K `K=50` raw boxes / recall / precision | `24.63` / `0.5023` / `0.2958` | exact/rounding OK |
| Top-K `K=50` NMS+cap boxes / recall / precision | `12.95` / `0.5009` / `0.5611` | exact/rounding OK |
| AITOD weighted AP25/AP50/AP75 raw | `0.6657` / `0.5734` / `0.1712` | exact match |
| AITOD weighted AP25/AP50/AP75 NMS+cap | `0.6929` / `0.5821` / `0.1695` | exact match |

Interpretation checked: AP is presented as secondary; the central gain is a review-burden trade-off.

## Scope Boundary Claims

Primary sources:

- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/cluster_size_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_image_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_sequence_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`
- `output/tables/localization_tier_boundary_summary.csv`

Verified rounded values:

| Claim location | Paper value | Evidence status |
|---|---:|---|
| AITOD block / parsed-sequence fixed-row pass | `0.1414` / `0.1352` | exact/rounding OK |
| AITOD parsed sequence mostly singleton | median `1`, p90 `1` | exact/rounding OK |
| UAVDT image pass | `0.1558` | exact/rounding OK |
| UAVDT block / sequence fail | `0.1803` / `0.2607` | exact/rounding OK |
| VisDrone image / sequence fail | `0.3173` / `0.4233` | exact/rounding OK |
| AITOD IoU 0.35 low-burden fixed-row image fail | `0.1725` | exact/rounding OK |
| IoU 0.50 unsupported under `alpha=0.16` | selected-row stresses fail | supported by localization-tier boundary summary |

## Metadata Invariant

AITOD-derived public result files were corrected to use `dataset=aitod` rather than the legacy `dataset=uavdt` value inherited from earlier UAVDT-oriented script tags. The metric values were not changed. The correction log is:

- `output/tables/metadata_corrections_audit.csv`

Some historical filenames still include `aitod_uavdt`; those are legacy contract/script tags. The `dataset` column inside released tables is authoritative.

## Safe Claim Boundary

Supported language:

- review-budget-constrained image-unit calibration;
- detector-specific per-cache operating point;
- AITOD RT-DETR positive review-burden result;
- AITOD-trained YOLO-family detector-specific positive row;
- direct fixed-threshold transfer failure;
- low-IoU object-presence triage;
- UAVDT/VisDrone, sequence-unit, and IoU-0.50 boundaries.

Unsupported language:

- detector-agnostic threshold;
- deployment-safety certificate;
- standard IoU-0.50 localization control;
- sequence-level guarantee;
- universal transfer across datasets or detectors.
