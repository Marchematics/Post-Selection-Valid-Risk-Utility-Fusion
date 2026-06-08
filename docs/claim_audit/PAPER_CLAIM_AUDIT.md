# Paper Claim Audit Report

**Date**: 2026-06-08  
**Audit basis**: manuscript values checked against included CSV outputs  
**Paper**: `Cluster-Aware Risk-Utility Audit for Low-IoU UAV Object-Presence Triage`  
**Overall Verdict**: PASS for checked quantitative claims in the submitted manuscript package.

## Bibliography Invariant

`paper/references.bib` was not edited. Last checked timestamp:

`2026-06-07 20:59:24.037793361 +0800`

## Claims Verified

### Main cluster audit

All Abstract and Table I values match the following fixed-candidate Hoeffding summary files for the main row `nms040_cap300@0.125`:

- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_image_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_sequence_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`

Verified rounded values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| Abstract/Table I AITOD image/block/sequence upper | 0.1496 / 0.1414 / 0.1352 | exact/rounding OK |
| Abstract/Table I AITOD precision and FP/image | 0.4886 and 12.03 | exact/rounding OK |
| Abstract/Table I UAVDT image upper, precision, FP/image | 0.1558, 0.4963, 27.43 | exact/rounding OK |
| Table I UAVDT block/sequence fail | 0.1803 / 0.2607 | exact/rounding OK |
| Table I VisDrone image/sequence fail | 0.3173 / 0.4233 | exact/rounding OK |

### AITOD design-sensitivity table

All Table II design-sensitivity values match:

- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_raw960_t0.0075_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_nms040_t0.02_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_nms040_cap300_t0.075_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_nms040_cap300_t0.15_image_operational_iou0.25_aitod_val_cache_summary.csv`

Verified values:

| Row | Paper values | Evidence status |
|---|---|---|
| raw960@0.0075 | upper 0.1285, FP/image 53.53, precision 0.1834 | exact/rounding OK |
| nms040@0.02 | upper 0.1188, FP/image 38.93 | exact/rounding OK |
| nms040_cap300@0.075 | upper 0.1331, precision 0.4036, FP/image 17.51 | exact/rounding OK |
| nms040_cap300@0.125 | upper 0.1496, precision 0.4886, FP/image 12.03 | exact/rounding OK |
| nms040_cap300@0.15 | upper 0.1597, precision 0.5241, FP/image 10.29 | exact/rounding OK as aggressive ablation, not main row |

### AITOD train-side consistency

The train-side consistency sentence in Section IV matches:

- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_train_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_train_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_train_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_train_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_train_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_train_cache_summary.csv`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| AITOD train image/block/sequence upper | 0.1238 / 0.1141 / 0.1114 | exact/rounding OK |
| AITOD train precision and FP/image | 0.5033 and 10.79 | exact/rounding OK |

The original train-side multi-unit selection diagnostic matches:

- `output/tables/multi_unit_train_selection_diagnostic_aitod_nmscap_t014_to_val.csv`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| Train-side tri-unit rule target | 0.14 for image/block/sequence | exact match |
| Selected threshold | 0.125 | exact match |
| Validation all-unit pass at main target | image/block/sequence pass at 0.16 | exact match |

The finite-family replay diagnostic matches:

- `scripts/aitod_multiunit_family_replay_fast.py`
- `output/tables/aitod_multiunit_family_replay_nms3_fast_selected.csv`
- `output/tables/aitod_multiunit_family_replay_nms3_fast_candidates.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.175_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.175_block_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.175_sequence_operational_iou0.25_aitod_val_cache_summary.csv`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| Declared replay size | 270 contract-threshold-unit tests | exact match: 3 contracts x 30 thresholds x 3 units |
| Strict replay target 0.14 selects | NMS+cap at 0.125 | exact match |
| Strict replay validation pass | image/block/sequence upper 0.1496 / 0.1414 / 0.1352, all pass at 0.16 | exact/rounding OK |
| Relaxed replay target 0.16 selects | NMS+cap at 0.175, train FP/image 7.83 | exact/rounding OK |
| Relaxed replay validation boundary | image/block upper 0.1696 / 0.1608 fail, sequence upper 0.1531 pass | exact/rounding OK |

### AITOD sample-size diagnostic

The Section IV sample-size diagnostic matches:

- `output/tables/fixed_unit_size_projection_aitod_val_nmscap0125_image_summary.csv`
- `output/tables/fixed_unit_size_projection_aitod_val_nmscap0125_block_summary.csv`
- `output/tables/fixed_unit_size_projection_aitod_val_nmscap0125_sequence_summary.csv`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| Required units if mean fixed | about 1000 image, 635 block, 464 sequence | exact/rounding OK |
| Available held-out units | 1869 image, 1663 block, 1464 sequence | exact/rounding OK |

### Utility table

Table III values match `output/tables/review_burden_simulation_v2.csv`.

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| AITOD raw and NMS+cap boxes/FP/miss/precision | 65.55/53.53/0.1710/0.1834 and 23.53/12.03/0.2072/0.4886 | exact/rounding OK |
| AITOD review-box reduction | 64.1% | rounding OK from 0.6410 |
| UAVDT rows | 37.77/12.07/0.1215/0.6804 and 54.46/27.43/0.0760/0.4963 | exact/rounding OK |
| UAVDT box increase | 44.2% | rounding OK from -0.4419 reduction field |
| VisDrone rows | 74.73/34.04/0.4246/0.5446 and 110.62/60.83/0.2960/0.4501 | exact/rounding OK |
| VisDrone box increase | 48.0% | arithmetic OK: 110.6241 / 74.7336 - 1 |

### Top-K review simulation

The AITOD human-review simulation sentence matches:

- `output/tables/topk_review_simulation.csv`
- `paper/figures/risk_utility_diagnostics.pdf`
- `scripts/plot_risk_utility_diagnostics.py`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| AITOD raw960 top-50 boxes/recall/precision | 24.63 / 0.5023 / 0.2958 | exact/rounding OK |
| AITOD NMS+cap top-50 boxes/recall/precision | 12.95 / 0.5009 / 0.5611 | exact/rounding OK |
| Figure risk-utility diagnostics | uses fixed-unit projection and top-K review CSVs | script regenerated successfully |

### AP diagnostics

AP claims match:

- `output/tables/ap_diagnostics_aitod_val_positive_summary.csv`
- `output/tables/ap_diagnostics_uavdt_test_positive_summary.csv`
- `output/tables/ap_diagnostics_visdrone_oracle_val_negative_summary.csv`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| AITOD weighted AP25/AP50/AP75 raw vs NMS+cap | 0.6657/0.5734/0.1712 vs 0.6929/0.5821/0.1695 | exact match |
| UAVDT weighted AP improves slightly but macro AP decreases | wAP25 0.8597 to 0.8738, wAP50 0.7999 to 0.8083; mAP25 0.7120 to 0.6861 | exact match |
| VisDrone AP improves but cluster risk fails | AP table confirms improvement; cluster summaries confirm failure | exact match |

### Localization-tier boundary diagnostics

Table IV values match:

- `output/tables/localization_tier_boundary_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.35_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.35_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.35_aitod_val_cache_summary.csv`
- `output/tables/cluster_contract_search_hoeffding_aitod_iou35_nms3_minloss_uavdt_image_operational_iou0.35_base_hoeffding_min_loss_aitod_train_to_val_nmscap_iou35_image_strict_fc_selected.csv`
- `output/tables/cluster_contract_search_hoeffding_aitod_iou35_nms3_minloss_uavdt_block_operational_iou0.35_base_hoeffding_min_loss_aitod_train_to_val_nmscap_iou35_block_strict_fc_selected.csv`
- `output/tables/cluster_contract_search_hoeffding_aitod_iou35_nms3_minloss_uavdt_sequence_operational_iou0.35_base_hoeffding_min_loss_aitod_train_to_val_nmscap_iou35_sequence_strict_fc_selected.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_uavdt_uavdt_nms040_cap300_t0.125_image_operational_iou0.35_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_visdrone_visdrone_nms040_cap300_t0.125_image_operational_iou0.35_visdrone_oracle_val_cache_summary.csv`
- `output/tables/cluster_contract_search_aitod_train_to_val_stress_uavdt_image_operational_iou0.5_base_eb_min_loss_aitod_train_to_val_nmscap_iou50_image_strict_fc_selected.csv`
- `output/tables/cluster_contract_search_aitod_train_to_val_stress_uavdt_sequence_operational_iou0.5_base_eb_min_loss_aitod_train_to_val_nmscap_iou50_sequence_strict_fc_selected.csv`
- `output/tables/cluster_contract_search_uavdt_trainval_to_test_stress_uavdt_image_operational_iou0.5_base_eb_min_loss_trainval_to_test_iou50_strict_fc_selected.csv`

Verified values:

| Claim location | Paper values | Evidence status |
|---|---|---|
| AITOD fixed IoU0.35 image/block/sequence | 0.1725 fail / 0.1636 fail / 0.1594 pass | exact/rounding OK |
| AITOD selected IoU0.35 image/block/sequence | 0.1386 / 0.1275 / 0.1222 pass; FP/image 39.25 or 61.25 | exact/rounding OK |
| UAVDT fixed IoU0.35 image | 0.1613 fail | exact/rounding OK |
| VisDrone fixed IoU0.35 image | 0.3264 fail | exact/rounding OK |
| AITOD train-to-val IoU0.50 image upper | 0.2184, fail | exact/rounding OK |
| AITOD train-to-val IoU0.50 sequence upper | 0.2047, fail | exact/rounding OK |
| UAVDT train+val-to-test IoU0.50 image upper | 0.1807, fail | exact/rounding OK |
| Boundary interpretation | IoU0.35 is supportable on AITOD only as higher-clutter min-loss evidence; IoU0.50 remains unsupported | supported by rows above |

## Scope Audit

Supported language:

- cluster-aware IoU-0.25 object-presence triage;
- AITOD image/block/sequence pass;
- UAVDT image-unit pass only;
- VisDrone and IoU-0.50 failures;
- review-burden reduction on AITOD only;
- `nms040_cap300@0.125` as the main row, with `0.15` reported only as a more aggressive near-boundary ablation.

Unsupported language searched and not found as positive claims:

- deployment-safety certificate;
- standard IoU-0.50 detection control;
- VisDrone success;
- UAVDT sequence-level control;
- globally optimal operating point;
- prospective lockbox certification.

## Reader Note

This file is a claim-to-result map for the submitted review package. It is intended to make the numerical evidence auditable from the included CSV summaries and scripts.
