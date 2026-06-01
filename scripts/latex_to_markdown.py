#!/usr/bin/env python3
"""Convert the current compact LaTeX manuscript to a readable Markdown draft."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
TITLE = "Post-Selection-Valid Risk-Utility Fusion for UAV Object Detection"
TITLE_SLUG = "Post-Selection-Valid_Risk-Utility_Fusion_for_UAV_Object_Detection"
OUT = PAPER / f"{TITLE_SLUG}.md"
FIGURE_COUNTER = 0


MACROS = {
    r"\Dcal": r"\mathcal{D}_{\mathrm{cal}}",
    r"\Deval": r"\mathcal{D}_{\mathrm{eval}}",
    r"\G": r"\mathcal{G}",
    r"\B": r"\mathcal{B}",
    r"\Tcal": r"\mathcal{T}",
    r"\risk": "R",
    r"\loss": r"\ell",
    r"\iou": r"\mathrm{IoU}",
    r"\cp": r"\mathrm{CP}",
    r"\ind": r"\mathbb{1}",
    r"\Prob": r"\mathbb{P}",
    r"\E": r"\mathbb{E}",
}


TABLE_CONTRACT = """**Table 1. Finite-family selection contract used in the main UAVDT audit.**

| Item | Fixed choice |
|---|---|
| Family | $M=10$ contracts listed in Sec. III; no hidden guard variants |
| Source support | Cross-resolution same-class support at IoU 0.45--0.50; fixed boosts/penalties |
| Risk/utility | Object miss risk $\\alpha=0.16$; clipped FP/image target $\\beta=150$ with $B=300$ |
| Correction | CP uses $\\eta_R=1-\\delta/(2M)$; FP uses $\\eta_F=1-\\delta/(2M|\\mathcal{T}|)$ |
| Selection | First select $t_c^\\star$ per candidate, then minimize FP upper bound across candidates |
| Groups | Sequence/title hash diagnostic; bad-group CP bound reported separately |
"""


TABLE_MAIN = """**Table 2. UAVDT image-hash certification: 237 calibration images/8872 objects and 68 evaluation images/2182 objects.** CP-U is the calibration miss-risk upper bound; Cal-FP is calibration clipped FP/image; FP-U is its Hoeffding upper bound.

| Selection | $t$ | Cal miss/$n$ | CP-U | Cal-FP | FP-U | Eval R | Prec. | FP/img |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw640 | .001 | 2298/8872 | 0.2668 | 163.8 | 187.7 | 0.2452 | 0.1484 | 139.0 |
| Raw960 | .020 | 1350/8872 | 0.1586 | 184.5 | 208.3 | 0.1402 | 0.1369 | 174.0 |
| Union | .069 | 1353/8872 | 0.1589 | 118.5 | 136.4 | 0.1444 | 0.2199 | 97.4 |
| NMS | .031 | 1348/8872 | 0.1583 | 96.0 | 119.9 | 0.1297 | 0.2466 | 85.3 |
| NMS+cap | .030 | 1350/8872 | 0.1586 | 98.2 | 122.0 | 0.1297 | 0.2423 | 87.3 |
| Utility | .040 | 1361/8872 | 0.1598 | 81.9 | 105.8 | 0.1421 | 0.2809 | 70.5 |
| Margin | .025 | 1266/8872 | 0.1489 | 107.5 | 131.4 | 0.1242 | 0.2260 | 96.3 |
| Family cert. | .030 | 1321/8872 | 0.1598 | 94.2 | 140.2 | 0.1338 | 0.2508 | 83.0 |
"""


TABLE_ABLATION = """**Table 3. Stress and transfer checks.** Status is empirical pass unless marked abstain; Seq-H denotes sequence-hash.

| Setting | $\\alpha$ | Risk | Prec. | FP/img | Status |
|---|---:|---:|---:|---:|---|
| UAVDT FP-min random mean | 0.16 | 0.1565 | 0.2820 | 74.2 | 4/5 |
| UAVDT margin random mean | 0.16 | 0.1460 | 0.2329 | 97.2 | 5/5 |
| UAVDT utility screen | 0.16 | 0.1421 | 0.2809 | 70.5 | pass |
| UAVDT family cert. | 0.16 | 0.1338 | 0.2508 | 83.0 | pass |
| UAVDT Seq-H raw | 0.16 | 0.2698 | 0.7419 | 13.6 | abstain |
| UAVDT Seq-support | 0.16 | 0.1545 | 0.2614 | 127.7 | obj/FP pass |
| VisDrone image-cert | 0.25 | 0.2354 | 0.3278 | 110.9 | pass |
| VisDrone seq. | 0.30 | 0.2960 | 0.5196 | 51.1 | pass |
| UAVDT IoU 0.50 cert. | 0.25 | 0.2269 | 0.2026 | 97.6 | pass |
"""


TABLE_SENSITIVITY = """**Table 4. Target and localization sensitivity for capped fusion.** Cert. and Pass are random-half counts.

| IoU | $\\alpha$ | Cert. | Pass | Risk | Prec. | FP/img |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.10 | 0/5 | 0/5 | 0.1157 | 0.1330 | 199.1 |
| 0.25 | 0.12 | 1/5 | 4/5 | 0.1157 | 0.1367 | 194.1 |
| 0.25 | 0.14 | 5/5 | 4/5 | 0.1320 | 0.2064 | 115.2 |
| 0.25 | 0.16 | 5/5 | 5/5 | 0.1474 | 0.2281 | 99.5 |
| 0.25 | 0.20 | 5/5 | 4/5 | 0.1973 | 0.4732 | 30.9 |
| 0.50 | 0.16 | 0/5 | 0/5 | 0.2393 | 0.1102 | 211.8 |
"""


TABLE_LOCKBOX = """**Table 4. Alpha and localization boundary card for the finite family.**

| Dataset/split | Contract | $\\alpha$ | Risk | FP/img | Status |
|---|---|---:|---:|---:|---|
| VisDrone image | union | 0.20 | 0.1895 | 531.1 | no feasible CP |
| VisDrone seq. | union | 0.20 | 0.2096 | 371.3 | fail |
| VisDrone image | NMS | 0.25 | 0.2354 | 110.9 | pass |
| VisDrone seq. | support | 0.25 | 0.2609 | 88.3 | fail |
| VisDrone image | cap, $\\tilde{\\alpha}=.29$ | 0.30 | 0.2742 | 57.1 | pass |
| VisDrone seq. | cap, $\\tilde{\\alpha}=.29$ | 0.30 | 0.2960 | 51.1 | pass |
| UAVDT image, IoU .50 | union | 0.25 | 0.2269 | 97.6 | pass |
| VisDrone image, IoU .50 | union | 0.30 | 0.2797 | 255.5 | risk only |
"""


def read_section(name: str) -> str:
    return (PAPER / "sections" / name).read_text(encoding="utf-8")


def strip_table_blocks(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        block = match.group(0)
        if "tab:contract" in block:
            table = TABLE_CONTRACT
        elif "tab:uavdt_main" in block:
            table = TABLE_MAIN
        elif "tab:stress" in block:
            table = TABLE_ABLATION
        elif "tab:boundary" in block:
            table = TABLE_LOCKBOX
        elif "tab:sensitivity" in block:
            table = TABLE_SENSITIVITY
        else:
            table = TABLE_MAIN
        return "\n\n" + table + "\n"

    text = re.sub(r"\\begin\{table\}(?:\[[^]]+\])?.*?\\end\{table\}", repl, text, flags=re.S)
    text = re.sub(r"\\begin\{table\*\}(?:\[[^]]+\])?.*?\\end\{table\*\}", repl, text, flags=re.S)
    return text


def convert_bibliography() -> str:
    bbl = (PAPER / "main.bbl").read_text(encoding="utf-8")
    items = re.split(r"\\bibitem\{([^}]+)\}", bbl)
    rows = ["## References\n"]
    for idx in range(1, len(items), 2):
        key = items[idx]
        body = items[idx + 1]
        body = re.split(r"\\bibitem|\n\\end\{thebibliography\}", body)[0]
        body = clean_latex(body).strip()
        body = re.sub(r"\s+", " ", body)
        rows.append(f"- [@{key}] {body}")
    return "\n".join(rows) + "\n"


def clean_latex(text: str) -> str:
    text = strip_table_blocks(text)

    for macro, value in MACROS.items():
        text = text.replace(macro, value)

    text = re.sub(r"\\section\{([^}]+)\}", r"## \1", text)
    text = re.sub(r"\\subsection\{([^}]+)\}", r"### \1", text)
    text = re.sub(r"\\paragraph\{([^}]+)\}", r"#### \1", text)
    text = re.sub(r"\\label\{[^}]+\}", "", text)
    text = re.sub(r"~\\ref\{([^}]+)\}", r" \\ref{\1}", text)
    text = re.sub(r"\\ref\{([^}]+)\}", r"`\1`", text)
    text = re.sub(r"\\cite\{([^}]+)\}", lambda m: "[" + "; ".join(f"@{x.strip()}" for x in m.group(1).split(",")) + "]", text)

    text = re.sub(r"\\begin\{equation\}\s*", "\n$$\n", text)
    text = re.sub(r"\s*\\end\{equation\}", "\n$$\n", text)

    text = re.sub(r"\\begin\{definition\}\[([^]]+)\]", r"\n> **Definition (\1).** ", text)
    text = re.sub(r"\\begin\{proposition\}\[([^]]+)\]", r"\n> **Proposition (\1).** ", text)
    text = re.sub(r"\\begin\{remark\}\[([^]]+)\]", r"\n> **Remark (\1).** ", text)
    text = text.replace(r"\begin{proof}", "\n> **Proof.** ")
    text = text.replace(r"\end{definition}", "\n")
    text = text.replace(r"\end{proposition}", "\n")
    text = text.replace(r"\end{remark}", "\n")
    text = text.replace(r"\end{proof}", "\n")

    text = text.replace(r"\begin{itemize}", "")
    text = text.replace(r"\end{itemize}", "")
    text = re.sub(r"\n\s*\\item\s+", "\n- ", text)

    text = re.sub(r"\\begin\{algorithm\}\[([^]]+)\]", r"\n> **Algorithm (\1).**\n", text)
    text = text.replace(r"\end{algorithm}", "\n")
    text = text.replace(r"\begin{enumerate}", "")
    text = text.replace(r"\end{enumerate}", "")
    text = re.sub(r"\\item\s+", "1. ", text)

    def fig_repl(match: re.Match[str]) -> str:
        global FIGURE_COUNTER
        block = match.group(0)
        path = re.search(r"\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}", block)
        path_text = path.group(1) if path else "figures/fig2_risk_utility_curve.pdf"
        if "fig1_selection_pipeline" in path_text:
            cap_text = (
                r"Finite-family selection pipeline. For each fixed contract, the CP rule selects $t_c^\star$ "
                r"with $\tilde{\alpha}\le\alpha$; the reported audit then chooses among candidate-level rows "
                r"by clipped FP upper bound and sends sequence-hash results to an empirical guard outside Theorem 1."
            )
        elif "fig_method_chain" in path_text:
            cap_text = (
                r"Main UAVDT image-hash certification result. Source-support selection improves the "
                r"miss-risk--precision--FP trade-off relative to raw 960 and NMS+cap."
            )
        elif "fig2_risk_utility_curve" in path_text:
            cap_text = (
                r"Raw UAVDT risk--utility curve. The $\alpha=0.16$ region requires a low threshold "
                r"and high FP burden before post-selection fusion."
            )
        else:
            cap = re.search(r"\\caption\{(.+?)\}", block, flags=re.S)
            cap_text = clean_latex(cap.group(1)).strip() if cap else ""
        FIGURE_COUNTER += 1
        return f"\n\n![{cap_text}]({path_text})\n\n*Figure {FIGURE_COUNTER}. {cap_text}*\n"

    text = re.sub(r"\\begin\{figure\}(?:\[[^]]+\])?.*?\\end\{figure\}", fig_repl, text, flags=re.S)

    text = re.sub(r"\\emph\{([^}]+)\}", r"*\1*", text)
    text = re.sub(r"\\texttt\{([^}]+)\}", r"`\1`", text)
    text = re.sub(r"\\mathrm\{([^}]+)\}", r"\\mathrm{\1}", text)
    text = text.replace(r"\toprule", "")
    text = text.replace(r"\midrule", "")
    text = text.replace(r"\bottomrule", "")
    text = text.replace(r"\newblock", "")
    text = text.replace(r"\hskip 1em plus 0.5em minus 0.4em\relax", " ")
    text = text.replace(r"\,", ",")
    text = text.replace("--", "--")

    text = text.replace(r"\begin{cases}", r"\begin{cases}")
    text = text.replace(r"\end{cases}", r"\end{cases}")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    title = f"# {TITLE}\n\n"
    authors = "**Authors:** Anonymous Authors\n\n"
    abstract = "## Abstract\n\n" + clean_latex(read_section("0_abstract.tex")) + "\n\n"
    keywords = (
        "**Keywords:** UAV aerial object detection; remote sensing; post-selection-valid risk-utility fusion; "
        "split calibration; Clopper--Pearson bounds.\n\n"
    )
    body = "\n\n".join(
        clean_latex(read_section(name))
        for name in [
            "1_introduction.tex",
            "2_related_work.tex",
            "3_method.tex",
            "4_experiments.tex",
            "5_discussion.tex",
            "6_conclusion.tex",
        ]
    )
    md = title + authors + abstract + keywords + body + "\n\n" + convert_bibliography()
    OUT.write_text(md + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
