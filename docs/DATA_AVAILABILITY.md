# Data and Code Availability Notes

The raw UAVDT and VisDrone images and annotations are third-party public benchmark data and are not redistributed in this supplement. The supplement includes derived tabular detector-cache data, split manifests, source code, result CSVs, and figure-generation code needed to audit the submitted results.

The derived cache tables use the following schema:

- `gt_rows.parquet`: `img_id`, `img_name`, `W`, `H`, `x1`, `y1`, `x2`, `y2`, `cls`, `is_ignore`, `is_small_lt24`, `is_tiny_lt16`.
- `pred_rows.parquet`: `img_id`, `img_name`, `resolution`, `x1`, `y1`, `x2`, `y2`, `score`, `cls`.
- `image_meta.csv`: `img_id`, `img_name`, `W`, `H`, `num_gt_valid`, `num_gt_ignore`, `num_small_lt24`, `num_tiny_lt16`, `ratio_small`.

Ready-to-paste manuscript wording if needed:

The UAVDT and VisDrone raw images and annotations are public third-party benchmark datasets. The supplementary material submitted with this Letter contains the derived detector-cache tables, split manifests, audit scripts, result CSVs, and figure-generation scripts needed to reproduce the reported finite-sample risk-utility audits. No raw UAV images are redistributed.

