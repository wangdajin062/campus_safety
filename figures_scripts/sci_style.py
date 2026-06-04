"""SCI-style matplotlib helpers (IEEE/Elsevier-friendly).

Font convention:
  - Text labels / titles / legends → Arial Rounded MT Bold (UI text)
  - Numeric tick labels          → Times New Roman (academic number style)
  - Use the helper function      → tnr_text() for inline numeric annotations
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Global defaults ──────────────────────────────────────────────────
mpl.rcParams.update({
    # Default UI text → Arial Rounded MT Bold
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial Rounded MT Bold", "DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
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


# ── Font helpers ─────────────────────────────────────────────────────

def tnr_text(ax, x, y, text, **kwargs):
    """Place *text* (typically a numeric annotation) using Times New Roman.

    Accepts all ``ax.text()`` keyword arguments; ``fontfamily`` is
    forced to ``"Times New Roman"`` regardless.
    """
    kwargs.setdefault("va", "center")
    kwargs.setdefault("fontsize", 10)
    ax.text(x, y, text, fontfamily="Arial Rounded MT Bold", **kwargs)


def save(fig, path, w=FIG_W_DOUBLE, h=FIG_H_DEFAULT):
    fig.set_size_inches(w, h)

    # Force all numerical tick labels → Times New Roman
    for ax in fig.axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily("Arial Rounded MT Bold")

    fig.savefig(path, dpi=420, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
