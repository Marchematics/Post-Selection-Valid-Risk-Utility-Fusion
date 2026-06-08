# Review-Budget-Constrained Calibration for Low-IoU UAV Object-Presence Triage

This is the standalone project directory for the GRSL-oriented manuscript:

**Review-Budget-Constrained Calibration for Low-IoU UAV Object-Presence Triage**

The current framing is a per-cache operating-point calibration protocol for frozen UAV detector outputs. A bounded miss-weighted loss alone can admit low-threshold cluttered rows because the FP term saturates. The manuscript therefore adds a hard image-unit review budget, FP/image <= 25, and reports a one-sided finite-sample loss bound on a disjoint AITOD validation audit.

Main result: on AITOD at IoU 0.25, the train-side conservative min-FP rule selects an RT-DETR-L 640+960 NMS+cap row at threshold 0.125. On validation it passes with `U_H=0.1496`, precision `0.4886`, and `12.03` FP/image, reducing review boxes by 64.1% relative to raw 960. The same fixed threshold does not transfer to an AITOD-trained YOLO-family cache, but the same budget-constrained procedure calibrates a detector-specific YOLO-family row at threshold 0.015 that passes with `U_H=0.1541` and `20.77` FP/image.

The supported claim is narrow: low-IoU image-review triage for frozen target caches. The manuscript does not claim detector-agnostic thresholds, deployment safety, sequence-level certification, source transfer, or IoU 0.50 localization control.

## Main Files

- `paper/main.tex`: canonical LaTeX entry point.
- `paper/main.pdf`: current compiled manuscript.
- `paper/PAPER_CLAIM_AUDIT.md`: claim-to-CSV audit for the active manuscript.
- `paper/REPRODUCIBILITY_CARD.md`: cache provenance, scripts, and reported outputs.
- `paper/IEEEtran.cls` and `paper/IEEEtran.bst`: local IEEEtran/GRSL-compatible template files.

## Recompile

From this directory:

```bash
cd paper
latexmk -pdf -file-line-error -interaction=nonstopmode -halt-on-error main.tex
```

The active manuscript compiles under `\documentclass[journal]{IEEEtran}` and is kept within the GRSL letter page limit.

## Layout

- `paper/`: manuscript source, compiled PDFs, template files, and audit cards.
- `paper/sections/`: LaTeX section files.
- `paper/figures/`: manuscript figures.
- `scripts/`: experiment and table-generation scripts used by the paper.
- `scripts/l1_clear_accept_evidence.py`: builds the review-budget calibration evidence tables.
- `scripts/build_aitod_yolo_family_cache.py`: builds the AITOD YOLO-family cache used for the second-detector stress test.
- `output/tables/`: result tables and audit CSVs.
- `submission_upload/`: staged GRSL submission bundles.

Some historical output filenames retain `aitod_uavdt` as a legacy contract/script tag from earlier cache routing. The `dataset` field inside released tables is authoritative; AITOD public tables have been corrected to `dataset=aitod`, with the repair listed in `output/tables/metadata_corrections_audit.csv`.
