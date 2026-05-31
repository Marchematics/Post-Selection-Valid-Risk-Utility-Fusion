from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#666666",
    "light_gray": "#D0D0D0",
}


def save_fig(fig: plt.Figure, name: str) -> None:
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    print(f"Saved: {out}")
