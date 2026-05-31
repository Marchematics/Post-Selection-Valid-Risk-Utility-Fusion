#!/usr/bin/env python3
"""Generate post-selection method-chain evidence line chart."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from paper_plot_style import COLORS


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent


def main() -> None:
    candidates = pd.read_csv(ROOT / "output/tables/post_selection_family_uavdt_a0.16_iou0.25_candidates.csv")
    family = pd.read_csv(ROOT / "output/tables/post_selection_family_uavdt_a0.16_iou0.25_family_selected.csv")
    margin = pd.read_csv(ROOT / "output/tables/post_selection_margin_uavdt_a0.16_sel0.151_iou0.25_selected.csv")
    image = candidates.loc[candidates["split"] == "image_lockbox"].set_index("contract")
    margin_image = margin.loc[margin["split"] == "image_lockbox"].iloc[0]
    rows = [
        ("Raw", image.loc["raw960"]),
        ("NMS+cap", image.loc["nms040_cap300"]),
        ("Utility-best\n$\\tilde{\\alpha}=.160$", image.loc["support_soft_a"]),
        ("Stability-best\n$\\tilde{\\alpha}=.151$", margin_image),
    ]
    data = pd.DataFrame(
        {
            "label": [name for name, _ in rows] + ["Family-\ncorr."],
            "risk": [row["eval_risk"] for _, row in rows]
            + [family.loc[family["split"] == "image_lockbox", "eval_risk"].iloc[0]],
            "precision": [row["eval_precision"] for _, row in rows]
            + [family.loc[family["split"] == "image_lockbox", "eval_precision"].iloc[0]],
            "fp_per_image": [row["eval_fp_img"] for _, row in rows]
            + [family.loc[family["split"] == "image_lockbox", "eval_fp_img"].iloc[0]],
        }
    )
    labels = data["label"].tolist()
    x = list(range(len(labels)))

    fig, ax = plt.subplots(1, 1, figsize=(4.15, 2.65))

    ax.plot(x, data["risk"], marker="o", ms=3.2, lw=1.25, color=COLORS["blue"], label="image-cert risk")
    ax.axhline(0.16, color=COLORS["gray"], ls="--", lw=0.9, label=r"$\alpha=.16$")
    ax.set_ylabel("miss risk")
    ax.set_ylim(0.118, 0.168)
    ax.set_xlim(-0.35, len(labels) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", rotation_mode="anchor", linespacing=1.05)
    ax.tick_params(axis="x", labelsize=5.2, pad=3)

    ax2 = ax.twinx()
    ax2.plot(x, data["precision"], marker="D", ms=3.0, lw=1.1, color=COLORS["red"], label="precision")
    ax2.set_ylabel("precision")
    ax2.set_ylim(0.12, 0.31)
    ax2.tick_params(axis="y", colors=COLORS["red"])
    ax2.yaxis.label.set_color(COLORS["red"])

    for i, fp in enumerate(data["fp_per_image"]):
        ax.text(
            i,
        0.1655,
            f"{fp:.0f} FP",
            ha="center",
            va="top",
            fontsize=6.5,
            color=COLORS["gray"],
        )

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + handles2,
        labels + labels2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        fontsize=6.2,
        columnspacing=0.9,
        handlelength=1.3,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.13, right=0.88, top=0.78, bottom=0.30)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_method_chain.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
