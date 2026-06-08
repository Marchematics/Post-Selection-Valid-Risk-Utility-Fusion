#!/usr/bin/env python3
"""Create compact review-budget diagnostics figures used by the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "output" / "tables"
FIG_DIR = ROOT / "paper" / "figures"


def load_curve(label: str) -> pd.DataFrame:
    path = TABLE_DIR / f"fixed_unit_size_projection_{label}_curve.csv"
    frame = pd.read_csv(path)
    frame["unit"] = label.rsplit("_", 1)[-1]
    return frame


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    curves = pd.concat(
        [
            load_curve("aitod_val_nmscap0125_image"),
            load_curve("aitod_val_nmscap0125_block"),
            load_curve("aitod_val_nmscap0125_sequence"),
        ],
        ignore_index=True,
    )
    topk = pd.read_csv(TABLE_DIR / "topk_review_simulation.csv")
    aitod = topk.loc[topk["label"].isin(["aitod_raw960_t0.0075", "aitod_nmscap_t0.125"])].copy()
    aitod["row"] = aitod["label"].map(
        {
            "aitod_raw960_t0.0075": "raw960",
            "aitod_nmscap_t0.125": "NMS+cap",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))

    ax = axes[0]
    styles = {
        "image": ("black", "-"),
        "block": ("0.35", "--"),
        "sequence": ("0.6", "-."),
    }
    for unit, group in curves.groupby("unit", sort=False):
        color, linestyle = styles[unit]
        ax.plot(group["subsample_n"], group["upper_mean"], color=color, linestyle=linestyle, linewidth=1.4, label=unit)
    ax.axhline(0.16, color="0.15", linewidth=0.9, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("cluster units")
    ax.set_ylabel("Hoeffding upper")
    ax.set_title("(a) AITOD cluster-size diagnostic")
    ax.set_ylim(0.12, 0.34)
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    for row, group in aitod.groupby("row", sort=False):
        style = "-" if row == "NMS+cap" else "--"
        color = "black" if row == "NMS+cap" else "0.55"
        ax.plot(group["top_k"], group["boxes_per_image"], color=color, linestyle=style, linewidth=1.5, marker="o", markersize=2.5, label=f"{row} boxes")
    ax.set_xlabel("top-K boxes/image cap")
    ax.set_ylabel("boxes/image")
    ax.set_title("(b) AITOD review-list trade-off")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax2 = ax.twinx()
    for row, group in aitod.groupby("row", sort=False):
        style = ":" if row == "NMS+cap" else "-."
        color = "black" if row == "NMS+cap" else "0.55"
        ax2.plot(group["top_k"], group["precision"], color=color, linestyle=style, linewidth=1.2, marker="s", markersize=2.2, label=f"{row} precision")
    ax2.set_ylabel("precision")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 300])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", ncol=1)

    fig.tight_layout(w_pad=1.2)
    out = FIG_DIR / "risk_utility_diagnostics.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")

    fig, ax = plt.subplots(1, 1, figsize=(3.45, 2.25))
    for row, group in aitod.groupby("row", sort=False):
        style = "-" if row == "NMS+cap" else "--"
        color = "black" if row == "NMS+cap" else "0.55"
        ax.plot(
            group["top_k"],
            group["boxes_per_image"],
            color=color,
            linestyle=style,
            linewidth=1.5,
            marker="o",
            markersize=2.5,
            label=f"{row} boxes",
        )
    ax.set_xlabel("top-K boxes/image cap")
    ax.set_ylabel("boxes/image")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax2 = ax.twinx()
    for row, group in aitod.groupby("row", sort=False):
        style = ":" if row == "NMS+cap" else "-."
        color = "black" if row == "NMS+cap" else "0.55"
        ax2.plot(
            group["top_k"],
            group["precision"],
            color=color,
            linestyle=style,
            linewidth=1.2,
            marker="s",
            markersize=2.2,
            label=f"{row} precision",
        )
    ax2.set_ylabel("precision")
    ax.set_xscale("log")
    ax.set_xticks([10, 25, 50, 100, 300])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="center right")
    fig.tight_layout()
    out = FIG_DIR / "review_budget_tradeoff.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
