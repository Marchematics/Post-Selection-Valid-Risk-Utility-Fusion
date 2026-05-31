# Source Data Map

This file maps manuscript displays to the source files in the supplement.

## Fig. 1

- Figure script: `paper/figures/gen_fig2_sequence_sensitivity.py`
- Generated figure: `paper/figures/fig_method_chain.pdf`
- Source CSVs:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_candidates.csv`
  - `output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_selected.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_selected.csv`

## Table I

The finite-family design is encoded in `scripts/post_selection_family_audit.py`, especially the `FAMILY` tuple and fixed audit constants.

## Table II

- Utility-best, NMS+cap, and raw rows:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_selected.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_candidates.csv`
- Stability-best rows:
  - `output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_selected.csv`
  - `output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_candidates.csv`
- Family-corrected rows:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_selected.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_candidates.csv`

## Table III

- UAVDT random/image/sequence stress:
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_selected.csv`
  - `output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_selected.csv`
  - `output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_selected.csv`
- VisDrone relaxed-target rows:
  - `output/tables/post_selection_family_visdrone_a0.25_iou0.25_selected.csv`
  - `output/tables/post_selection_family_guarded_select_visdrone_a0.29_iou0.25_selected.csv`
- UAVDT IoU 0.50 row:
  - `output/tables/post_selection_family_iou_uavdt_a0.25_iou0.5_selected.csv`

## Table IV

- VisDrone alpha and localization boundary rows:
  - `output/tables/post_selection_family_visdrone_a0.2_iou0.25_selected.csv`
  - `output/tables/post_selection_family_visdrone_a0.25_iou0.25_selected.csv`
  - `output/tables/post_selection_family_visdrone_a0.3_iou0.25_selected.csv`
  - `output/tables/post_selection_family_guarded_select_visdrone_a0.29_iou0.25_selected.csv`
  - `output/tables/post_selection_family_iou_visdrone_a0.3_iou0.5_selected.csv`

## Split Manifests

Split manifests are in:

- `data/manifests/uavdt/`
- `data/manifests/visdrone/`
- image-hash manifests stored with result tables:
  - `output/tables/uavdt_revision_image_lockbox_cal.csv`
  - `output/tables/uavdt_revision_image_lockbox_eval.csv`
  - `output/tables/visdrone_revision_image_lockbox_cal.csv`
  - `output/tables/visdrone_revision_image_lockbox_eval.csv`

