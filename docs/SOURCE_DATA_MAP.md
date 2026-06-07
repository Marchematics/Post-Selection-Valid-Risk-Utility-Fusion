# Source Data Map

This file maps manuscript displays to the source files in the supplement.

## Fig. 1

- Figure script: `paper/figures/gen_fig2_sequence_sensitivity.py`
- Generated figure: `paper/figures/fig_method_chain.pdf`
- Source CSVs:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_candidates.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_selected.csv`
  - `output/tables/cluster_ratio_trainval_imagehash90_full_uavdt_image_a0.16_iou0.25_base_eb_selected.csv`
  - `output/tables/cluster_ratio_trainval_seqhash90_probe_uavdt_sequence_a0.16_iou0.25_base_eb_selected.csv`

## Table I

- Raw960 and family object-level rows:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_candidates.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_selected.csv`
- Validation-only cluster-ratio abstention:
  - `output/tables/cluster_ratio_valonly_imagehash_nmax_uavdt_image_a0.16_iou0.25_base_eb_bounds.csv`
  - `output/tables/cluster_ratio_valonly_imagehash_nmax_uavdt_image_a0.16_iou0.25_base_eb_selected.csv`
- Train+val cluster-ratio development stress:
  - `output/tables/cluster_ratio_trainval_imagehash90_full_uavdt_image_a0.16_iou0.25_base_eb_bounds.csv`
  - `output/tables/cluster_ratio_trainval_imagehash90_full_uavdt_image_a0.16_iou0.25_base_eb_selected.csv`
- Sequence-ratio abstention:
  - `output/tables/cluster_ratio_trainval_seqhash90_probe_uavdt_sequence_a0.16_iou0.25_base_eb_bounds.csv`
  - `output/tables/cluster_ratio_trainval_seqhash90_probe_uavdt_sequence_a0.16_iou0.25_base_eb_selected.csv`

## Table II

- Validation utility and margin screens:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_selected.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_candidates.csv`
  - `output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_selected.csv`
  - `output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_candidates.csv`
- UAVDT sequence-support and sequence-ratio rows:
  - `output/tables/cluster_ratio_trainval_seqhash90_probe_uavdt_sequence_a0.16_iou0.25_base_eb_selected.csv`
- VisDrone transfer rows:
  - `output/tables/post_selection_family_visdrone_a0.25_iou0.25_selected.csv`
  - `output/tables/post_selection_family_guarded_select_visdrone_a0.29_iou0.25_selected.csv`
- UAVDT IoU 0.50 localization boundary:
  - `output/tables/post_selection_family_iou_uavdt_a0.25_iou0.5_selected.csv`

## Split Manifests

Split manifests are in:

- `data/manifests/uavdt/`
- `data/manifests/visdrone/`
- image-hash manifests stored with result tables:
  - `output/tables/uavdt_revision_image_lockbox_cal.csv`
  - `output/tables/uavdt_revision_image_lockbox_eval.csv`
  - `output/tables/visdrone_revision_image_lockbox_cal.csv`
  - `output/tables/visdrone_revision_image_lockbox_eval.csv`
