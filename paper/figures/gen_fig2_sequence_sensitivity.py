#!/usr/bin/env python3
"""Generate the main UAVDT result chart used as Fig. 1."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from paper_plot_style import COLORS


FIG_DIR = Path(__file__).resolve().parent


def main() -> None:
    data = pd.DataFrame(
        {
            "label": [
                "Raw960 val\nobject",
                "Family val\nobject",
                "Train+val\ncluster",
                "Sequence\nabstain",
            ],
            "risk": [0.1402, 0.1338, 0.0306, 0.0024],
            "precision": [0.1369, 0.2508, 0.3741, 0.2291],
            "fp_per_image": [174.0, 83.0, 36.9, 93.7],
        }
    )
    x = list(range(len(data)))

    fig, ax = plt.subplots(1, 1, figsize=(4.15, 2.45))

    main_x = [0, 1, np.nan, 3]
    main_risk = [data.loc[0, "risk"], data.loc[1, "risk"], np.nan, data.loc[3, "risk"]]
    main_precision = [data.loc[0, "precision"], data.loc[1, "precision"], np.nan, data.loc[3, "precision"]]

    ax.plot(main_x, main_risk, marker="o", ms=3.6, lw=1.35, color=COLORS["blue"], label="eval miss risk")
    ax.scatter(
        [2],
        [data.loc[2, "risk"]],
        marker="o",
        s=16,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.9,
        zorder=3,
        label="dev stress",
    )
    ax.axhline(0.16, color=COLORS["gray"], ls="--", lw=0.9, label=r"$\alpha=.16$")
    ax.set_ylabel("miss risk")
    ax.set_ylim(-0.012, 0.175)
    ax.set_xlim(-0.35, len(data) - 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"].tolist(), rotation=0, ha="center", linespacing=1.0)
    ax.tick_params(axis="x", labelsize=6.0, pad=2)

    ax2 = ax.twinx()
    ax2.plot(main_x, main_precision, marker="D", ms=3.2, lw=1.2, color=COLORS["red"], label="precision")
    ax2.scatter(
        [2],
        [data.loc[2, "precision"]],
        marker="D",
        s=16,
        facecolors="white",
        edgecolors=COLORS["gray"],
        linewidths=0.9,
        zorder=3,
    )
    ax2.set_ylabel("precision")
    ax2.set_ylim(0.08, 0.42)
    ax2.tick_params(axis="y", colors=COLORS["red"])
    ax2.yaxis.label.set_color(COLORS["red"])

    for i, fp in enumerate(data["fp_per_image"]):
        ax.text(
            i,
            0.006,
            f"{fp:.0f} FP/img",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color=COLORS["gray"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.3},
        )

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles + handles2,
        labels + labels2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        fontsize=6.2,
        columnspacing=0.8,
        handlelength=1.2,
        handletextpad=0.35,
    )

    fig.subplots_adjust(left=0.13, right=0.88, top=0.77, bottom=0.24)
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_method_chain.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.025)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
