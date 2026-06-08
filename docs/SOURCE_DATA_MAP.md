# Source Data Map

This file maps the submitted GRSL manuscript displays and quantitative claims
to repository files.

Some historical output filenames retain `aitod_uavdt` as a legacy
contract/script tag from earlier cache routing. The `dataset` field inside each
released table is authoritative; AITOD public tables have been corrected to
`dataset=aitod`, with the repair listed in
`output/tables/metadata_corrections_audit.csv`.

## Main Cluster Audit

Manuscript Table I and abstract values:

- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_uavdt_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_uavdt_test_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_image_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_visdrone_visdrone_nms040_cap300_t0.125_sequence_operational_iou0.25_visdrone_oracle_val_cache_summary.csv`

## AITOD Train-Side Replay and Sensitivity

- `output/tables/aitod_multiunit_family_replay_nms3_fast_selected.csv`
- `output/tables/aitod_multiunit_family_replay_nms3_fast_candidates.csv`
- `output/tables/multi_unit_train_selection_diagnostic_aitod_nmscap_t014_to_val.csv`
- `output/tables/cluster_size_summary.csv`
- `output/tables/aitod_loss_cap_alpha_sensitivity.csv`
- `output/tables/metadata_corrections_audit.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_train_uavdt_nms040_cap300_t0.125_image_operational_iou0.25_aitod_train_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_train_uavdt_nms040_cap300_t0.125_block_operational_iou0.25_aitod_train_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_train_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.25_aitod_train_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_raw960_t0.0075_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_nms040_t0.02_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_nms040_cap300_t0.075_image_operational_iou0.25_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_aitod_ablation_uavdt_nms040_cap300_t0.15_image_operational_iou0.25_aitod_val_cache_summary.csv`

## Figure and Review-Burden Diagnostics

- Figure: `paper/figures/risk_utility_diagnostics.pdf`
- Figure script: `scripts/plot_risk_utility_diagnostics.py`
- Sample-size CSVs:
  - `output/tables/fixed_unit_size_projection_aitod_val_nmscap0125_image_summary.csv`
  - `output/tables/fixed_unit_size_projection_aitod_val_nmscap0125_block_summary.csv`
  - `output/tables/fixed_unit_size_projection_aitod_val_nmscap0125_sequence_summary.csv`
- Review burden: `output/tables/review_burden_simulation_v2.csv`
- Top-K review simulation: `output/tables/topk_review_simulation.csv`
- AP diagnostics:
  - `output/tables/ap_diagnostics_aitod_val_positive_summary.csv`
  - `output/tables/ap_diagnostics_uavdt_test_positive_summary.csv`
  - `output/tables/ap_diagnostics_visdrone_oracle_val_negative_summary.csv`

## Localization-Tier Boundaries

- `output/tables/localization_tier_boundary_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_image_operational_iou0.35_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_block_operational_iou0.35_aitod_val_cache_summary.csv`
- `output/tables/fixed_cluster_candidate_eval_hoeffding_iou35_aitod_uavdt_nms040_cap300_t0.125_sequence_operational_iou0.35_aitod_val_cache_summary.csv`
- `output/tables/cluster_contract_search_aitod_train_to_val_stress_uavdt_image_operational_iou0.5_base_eb_min_loss_aitod_train_to_val_nmscap_iou50_image_strict_fc_selected.csv`
- `output/tables/cluster_contract_search_aitod_train_to_val_stress_uavdt_sequence_operational_iou0.5_base_eb_min_loss_aitod_train_to_val_nmscap_iou50_sequence_strict_fc_selected.csv`
- `output/tables/cluster_contract_search_uavdt_trainval_to_test_stress_uavdt_image_operational_iou0.5_base_eb_min_loss_trainval_to_test_iou50_strict_fc_selected.csv`

## Full Claim Map

See:

- `docs/claim_audit/PAPER_CLAIM_AUDIT.md`
- `docs/REPRODUCIBILITY_CARD.md`
