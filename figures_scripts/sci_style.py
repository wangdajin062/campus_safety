"""SCI-style matplotlib helpers (IEEE/Elsevier-friendly)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

# Use professional defaults aligned with IEEE / Elsevier
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette: colorblind-safe, journal-grade
PALETTE = {
    "primary":   "#1f77b4",   # main result
    "secondary": "#d62728",   # baseline
    "tertiary":  "#2ca02c",   # auxiliary
    "neutral":   "#7f7f7f",   # ablation / reference
    "highlight": "#ff7f0e",   # emphasis
    "purple":    "#9467bd",
    "teal":      "#17becf",
    "brown":     "#8c564b",
}

FIG_W_SINGLE = 3.5     # single-column
FIG_W_DOUBLE = 7.16    # double-column
FIG_H_DEFAULT = 2.5


def save(fig, path, w=FIG_W_DOUBLE, h=FIG_H_DEFAULT):
    fig.set_size_inches(w, h)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
