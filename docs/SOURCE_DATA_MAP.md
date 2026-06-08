# Source Data Map

This file maps the active GRSL manuscript displays and quantitative claims to
repository files.

Some historical output filenames retain `aitod_uavdt` as a legacy
contract/script tag from earlier cache routing. The `dataset` field inside each
released table is authoritative; AITOD public tables have been corrected to
`dataset=aitod`, with the repair listed in
`output/tables/metadata_corrections_audit.csv`.

## Main Budget-Constrained Calibration Table

Manuscript Table I and abstract values:

- `output/tables/grsl_budget_constrained_main_summary.csv`
- `output/tables/l1_aitod_yolo11n_fixed_aitod_nms040_cap300_t0.125_image_operational_iou0.25_sequence_summary.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_t013_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_t014_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`
- `output/tables/l1_aitod_yolo11n_train_to_val_search_aitod_image_operational_iou0.25_base_hoeffding_min_fp_val_strict_selected.csv`

## Review Burden and AP Diagnostics

Manuscript Table II and Fig. 1:

- Figure: `paper/figures/review_budget_tradeoff.pdf`
- Figure script: `scripts/plot_risk_utility_diagnostics.py`
- Review burden: `output/tables/review_burden_simulation_v2.csv`
- Top-K review simulation: `output/tables/topk_review_simulation.csv`
- AP diagnostics:
  - `output/tables/ap_diagnostics_aitod_val_positive_summary.csv`
  - `output/tables/ap_diagnostics_uavdt_test_positive_summary.csv`
  - `output/tables/ap_diagnostics_visdrone_oracle_val_negative_summary.csv`

## Boundary Diagnostics

Scope and limitation claims:

- `output/tables/cluster_size_summary.csv`
- `output/tables/localization_tier_boundary_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_image_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_sequence_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.35_aitod_val_cache_summary.csv`

## Scripts

- `scripts/l1_clear_accept_evidence.py`
- `scripts/build_aitod_yolo_family_cache.py`
- `scripts/fixed_cluster_candidate_eval.py`
- `scripts/cluster_contract_search_fast.py`
- `scripts/review_burden_simulation.py`
- `scripts/topk_review_simulation.py`
- `scripts/general_detection_ap_diagnostics.py`
- `scripts/plot_risk_utility_diagnostics.py`

## Full Claim Map

See:

- `docs/claim_audit/PAPER_CLAIM_AUDIT.md`
- `docs/REPRODUCIBILITY_CARD.md`
