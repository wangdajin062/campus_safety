"""
Figure 8: 30-day IRB deployment results.

Data source: safety_data.py (LATENCY_*, DEPLOYMENT, HEAD_TO_HEAD).
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
from sci_style import tnr_text
from safety_data import (
    LATENCY_COMPONENTS, LATENCY_P50_MS, LATENCY_P99_MS,
    DEPLOYMENT, HEAD_TO_HEAD,
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.4, 3.0),
                                     gridspec_kw={"wspace": 0.55})

# ---- (a) Latency decomposition ----
x = np.arange(2)
bottom_p50 = 0
bottom_p99 = 0
colors = ["#2ca02c", "#9467bd", "#ff7f0e", "#1f77b4"]
for i, (c, lat50, lat99) in enumerate(
        zip(LATENCY_COMPONENTS, LATENCY_P50_MS, LATENCY_P99_MS)):
    ax1.bar([0], [lat50], bottom=bottom_p50, color=colors[i],
            edgecolor="black", lw=0.5, label=c)
    ax1.bar([1], [lat99], bottom=bottom_p99, color=colors[i],
            edgecolor="black", lw=0.5)
    if lat50 > 25:
        tnr_text(ax1, 0, bottom_p50 + lat50/2, f"{lat50}",
                 ha="center", color="white", weight="bold")
    if lat99 > 30:
        tnr_text(ax1, 1, bottom_p99 + lat99/2, f"{lat99}",
                 ha="center", color="white", weight="bold")
    bottom_p50 += lat50
    bottom_p99 += lat99

tnr_text(ax1, 0, bottom_p50 + 8, f"{bottom_p50} ms",
         ha="center", color="#222", weight="bold")
# Move 342 ms label well above the 350-target line to avoid overlap
tnr_text(ax1, 1, bottom_p99 + 30, f"{bottom_p99} ms",
         ha="center", color="#cc5500", weight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(["P50", "P99"])
ax1.set_ylabel("Latency (ms)")
ax1.set_ylim(0, 430)
ax1.axhline(350, color="#cc5500", lw=0.7, ls="--", alpha=0.6)
ax1.text(-0.4, 358, "350 ms target",
         fontsize=6, ha="left", color="#cc5500", style="italic")
ax1.set_title("(a) Latency (SD 8 Gen 3)", weight="bold", fontsize=9)
ax1.legend(loc="upper left", fontsize=5.8, ncol=2,
           handlelength=1, columnspacing=0.7,
           bbox_to_anchor=(-0.02, 0.78), frameon=False)

# ---- (b) Operational metrics ----
metrics = ["Precision", "Recall", "Satisfact.\n(scaled)"]
vals    = [DEPLOYMENT["precision"], DEPLOYMENT["recall"],
           DEPLOYMENT["satisfaction_pct"]]
# Symmetric half-width of 95% CI for precision/recall, ±1.5 for satisfaction
err = [
    (DEPLOYMENT["precision_ci"][1] - DEPLOYMENT["precision_ci"][0]) / 2,
    (DEPLOYMENT["recall_ci"][1]    - DEPLOYMENT["recall_ci"][0])    / 2,
    1.5,
]

x2 = np.arange(len(metrics))
colors_m = ["#1f77b4", "#2ca02c", "#ff7f0e"]
ax2.bar(x2, vals, yerr=err, color=colors_m, edgecolor="black",
        lw=0.6, capsize=4, error_kw={"elinewidth": 0.8})
for i, (v, e) in enumerate(zip(vals, err)):
    tnr_text(ax2, i, v + e + 0.6, f"{v:.1f}%",
             ha="center", weight="bold")

ax2.set_xticks(x2)
ax2.set_xticklabels(metrics, fontsize=7.5)
ax2.set_ylabel("Score (%)")
ax2.set_ylim(85, 104)
n_students = DEPLOYMENT["n_students"]
n_days = DEPLOYMENT["duration_days"]
ax2.set_title(f"(b) {n_days}-d ops, $n=" + f"{n_students:,}".replace(",", "\\,") + "$",
              weight="bold", fontsize=9)
ax2.text(1, 86, f"{DEPLOYMENT['irb_id']} • $p<10^{{-3}}$",
         ha="center", fontsize=6.2, color="#444", style="italic")

# ---- (c) Head-to-head vs SAFE-QAQ — separate metric panels ----
ours = HEAD_TO_HEAD["ours"]
safe = HEAD_TO_HEAD["safe_qaq"]

# Three rows of horizontal bars, each on its own normalised scale
ax3.set_title("(c) vs SAFE-QAQ [27]", weight="bold", fontsize=9)
ax3.set_xlim(0, 1.0)
ax3.set_ylim(-0.5, 5.5)
ax3.set_xticks([])
ax3.set_yticks([])
for spine in ("top", "right", "bottom", "left"):
    ax3.spines[spine].set_visible(False)

# Row data: (label, ours_value, safe_value, ours_str, safe_str, our_better_str)
rows = [
    ("F1 score",       ours["f1"], safe["f1"],
     f"{ours['f1']:.3f}", f"{safe['f1']:.3f}",
     f"+{(ours['f1']-safe['f1'])*100:.1f}%"),
    ("Model size",     ours["size_mb"], safe["size_mb"],
     f"{ours['size_mb']} MB", f"{safe['size_mb']} MB",
     f"{safe['size_mb']/ours['size_mb']:.1f}× smaller"),
    ("Median latency", ours["latency_ms"], safe["latency_ms"],
     f"{ours['latency_ms']} ms", f"{safe['latency_ms']} ms",
     f"{safe['latency_ms']/ours['latency_ms']:.1f}× faster"),
]

for i, (label, ours_v, safe_v, ours_str, safe_str, better) in enumerate(rows):
    y_pair = 4.5 - i * 2.0
    # Normalise to [0, 1]: smaller value = shorter bar (for size/latency we
    # want "smaller is better", so use ratio relative to the larger one).
    max_v = max(ours_v, safe_v)
    ours_norm = ours_v / max_v
    safe_norm = safe_v / max_v
    # Section title above
    ax3.text(0, y_pair + 0.85, label, fontsize=7.5, weight="bold", color="#222")
    # Ours bar
    ax3.barh(y_pair + 0.30, ours_norm * 0.55, height=0.40, left=0.05,
             color="#ff7f0e", edgecolor="#cc5500", lw=0.8)
    tnr_text(ax3, 0.05 + ours_norm * 0.55 + 0.02, y_pair + 0.30, ours_str,
             va="center", color="#cc5500", weight="bold")
    ax3.text(0.02, y_pair + 0.30, "Ours", va="center", ha="right",
             fontsize=6, color="#cc5500")
    # SAFE-QAQ bar
    ax3.barh(y_pair - 0.30, safe_norm * 0.55, height=0.40, left=0.05,
             color="#7f7f7f", edgecolor="black", lw=0.5)
    tnr_text(ax3, 0.05 + safe_norm * 0.55 + 0.02, y_pair - 0.30, safe_str,
             va="center", color="#222")
    ax3.text(0.02, y_pair - 0.30, "[27]", va="center", ha="right",
             fontsize=6, color="#666")
    # "Better" annotation on the right
    tnr_text(ax3, 0.97, y_pair, better, ha="right", va="center",
             color="#1e5a23", weight="bold",
             bbox=dict(facecolor="#E8F5E9", edgecolor="#1e5a23",
                       lw=0.5, pad=2))

sci.save(fig, "fig08_deployment.png", w=7.4, h=3.0)
print("Saved fig08_deployment.png")
