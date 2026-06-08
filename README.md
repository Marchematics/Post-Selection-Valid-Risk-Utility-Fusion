# Cluster-Aware Risk Audit for Low-IoU UAV Object-Presence Triage

This is the standalone project directory for the GRSL-oriented manuscript:

**Cluster-Aware Risk Audit for Low-IoU UAV Object-Presence Triage**

The directory was split out from the broader UAV risk-certification workspace so this article is no longer mixed with other paper drafts or experiments.

Current framing: this is a post-hoc cluster-aware operating-point audit for frozen UAV detector caches, not a deployment-safety certification or a prospective lockbox study. The main AITOD validation row is a fixed 640+960 NMS+cap contract at threshold 0.125 and IoU 0.25. It passes the declared AITOD image/block/parsed-sequence audits at alpha 0.16, reduces review boxes by 64.1% relative to raw RT-DETR-L/960, and reports the increased miss risk. UAVDT passes only image units, VisDrone fails, and IoU 0.50 remains unsupported.

## Main Files

- `paper/main.tex`: canonical LaTeX entry point.
- `paper/main.pdf`: current compiled manuscript.
- `paper/IEEEtran.cls` and `paper/IEEEtran.bst`: local IEEEtran/GRSL-compatible template files.
- `paper/template_refs/grsl/`: downloaded IEEE/GRSL template references and checksums.

## Recompile

From this directory:

```bash
cd paper
latexmk -pdf -file-line-error -interaction=nonstopmode -halt-on-error main.tex
```

The manuscript currently compiles to 5 pages under `\documentclass[journal]{IEEEtran}`, matching the GRSL letter length constraint.

The active PDF is aligned to the current GRSL scope: retrospective fixed-row audit, AITOD positive evidence, UAVDT/VisDrone/IoU boundary failures, metadata-corrected AITOD result tables, and explicit low-IoU triage framing.

## Layout

- `paper/`: manuscript source, compiled PDFs, template files, audit cards.
- `paper/sections/`: LaTeX section files.
- `paper/figures/`: local figure files copied into this standalone package.
- `scripts/`: experiment and table-generation scripts used by the paper.
- `scripts/latex_to_markdown.py`: reproducible LaTeX-to-Markdown conversion script for the current article.
- `output/tables/`: result tables and audit CSVs.
- `output/review/`: review and execution reports.
- `notes/`: project notes and handoff material.
- `templates/`: downloaded IEEEtran/GRSL template archive.
