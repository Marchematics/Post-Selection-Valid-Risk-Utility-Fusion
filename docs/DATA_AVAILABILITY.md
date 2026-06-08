# Data and Code Availability

Code, split manifests when available, selected result CSVs, claim-audit notes,
and figure-generation scripts for the GRSL manuscript are available in this
repository:

`https://github.com/Marchematics/Post-Selection-Valid-Risk-Utility-Fusion`

The raw UAV benchmark images and annotations are public third-party datasets
and are not redistributed here. Some audit scripts require derived detector
caches with the following schema:

- `gt_rows.parquet`: `img_id`, `img_name`, `W`, `H`, `x1`, `y1`, `x2`, `y2`,
  `cls`, `is_ignore`, `is_small_lt24`, `is_tiny_lt16`.
- `pred_rows.parquet`: `img_id`, `img_name`, `resolution`, `x1`, `y1`, `x2`,
  `y2`, `score`, `cls`.
- `image_meta.csv`: `img_id`, `img_name`, `W`, `H`, `num_gt_valid`,
  `num_gt_ignore`, `num_small_lt24`, `num_tiny_lt16`, `ratio_small`.

The submitted supplementary review package contains the same selected scripts,
CSV summaries, claim map, and generated figure used to audit the manuscript
tables.

Ready-to-paste manuscript wording:

> Code, split manifests, result CSVs, and figure-generation scripts are
> available at
> `https://github.com/Marchematics/Post-Selection-Valid-Risk-Utility-Fusion`.
> The same materials are included in the supplementary review package. Raw
> benchmark images are public third-party data and are not redistributed.
