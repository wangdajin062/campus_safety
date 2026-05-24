"""
Figure 8: 30-day IRB deployment results.

Data source: safety_data.py (LATENCY_*, DEPLOYMENT, HEAD_TO_HEAD).
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
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
        ax1.text(0, bottom_p50 + lat50/2, f"{lat50}", ha="center",
                 va="center", fontsize=7, color="white", weight="bold")
    if lat99 > 30:
        ax1.text(1, bottom_p99 + lat99/2, f"{lat99}", ha="center",
                 va="center", fontsize=7, color="white", weight="bold")
    bottom_p50 += lat50
    bottom_p99 += lat99

ax1.text(0, bottom_p50 + 8, f"{bottom_p50} ms",
         ha="center", fontsize=7, weight="bold", color="#222")
ax1.text(1, bottom_p99 + 8, f"{bottom_p99} ms",
         ha="center", fontsize=7, weight="bold", color="#cc5500")

ax1.set_xticks(x)
ax1.set_xticklabels(["P50", "P99"])
ax1.set_ylabel("Latency (ms)")
ax1.set_ylim(0, 430)
ax1.axhline(350, color="#cc5500", lw=0.7, ls="--", alpha=0.6)
ax1.text(1.45, 320, "P99 target\n< 350 ms", fontsize=6.0,
         ha="right", color="#cc5500", style="italic")
ax1.set_title("(a) Latency (SD 8 Gen 3)", weight="bold", fontsize=9)
ax1.legend(loc="upper center", fontsize=6, ncol=2,
           handlelength=1, columnspacing=0.7, bbox_to_anchor=(0.5, 0.98))

# ---- (b) Operational metrics ----
metrics = ["Precision", "Recall", "Satisfaction\n(scaled)"]
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
    ax2.text(i, v + e + 0.6, f"{v:.1f}%", ha="center",
             fontsize=7.5, weight="bold")

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

# ---- (c) Head-to-head vs SAFE-QAQ on log y ----
ours = HEAD_TO_HEAD["ours"]
safe = HEAD_TO_HEAD["safe_qaq"]
metrics3 = ["F1\nratio", "Size\nratio", "Latency\nratio"]
ours_norm = [1.0, 1.0, 1.0]
safe_norm = [
    safe["f1"] / ours["f1"],
    safe["size_mb"] / ours["size_mb"],
    safe["latency_ms"] / ours["latency_ms"],
]

x3 = np.arange(len(metrics3))
w = 0.32
ax3.bar(x3 - w/2, ours_norm, w, color="#ff7f0e", edgecolor="#cc5500",
        lw=1.2, label="Ours")
ax3.bar(x3 + w/2, safe_norm, w, color="#7f7f7f", edgecolor="black",
        lw=0.5, label="SAFE-QAQ")

ax3.set_xticks(x3)
ax3.set_xticklabels(metrics3, fontsize=7.5)
ax3.set_ylabel("Ratio relative to ours")
ax3.set_yscale("log")
ax3.set_ylim(0.5, 80)
ax3.set_title("(c) vs SAFE-QAQ [27]", weight="bold", fontsize=9)
ax3.legend(loc="lower right", fontsize=7, frameon=True)
ax3.axhline(1.0, color="#888", lw=0.5, ls=":", alpha=0.6)

ratios_txt = [
    f"F1 {(ours['f1']/safe['f1']-1)*100:+.1f}%",
    f"{safe_norm[1]:.1f}× smaller",
    f"{safe_norm[2]:.1f}× faster",
]
ours_abs = [f"{ours['f1']:.3f}",
            f"{ours['size_mb']} MB",
            f"{ours['latency_ms']} ms"]
safe_abs = [f"{safe['f1']:.3f}",
            f"{safe['size_mb']} MB",
            f"{safe['latency_ms']} ms"]

for i in range(3):
    ax3.text(x3[i] - w/2, ours_norm[i] * 0.72, ours_abs[i],
             ha="center", va="top", fontsize=5.8,
             color="#cc5500", weight="bold")
    sy = safe_norm[i] * 0.72 if safe_norm[i] > 1.5 else safe_norm[i] * 1.18
    ax3.text(x3[i] + w/2, sy, safe_abs[i],
             ha="center",
             va="top" if safe_norm[i] > 1.5 else "bottom",
             fontsize=5.8, color="#222")
    y_lab = max(ours_norm[i], safe_norm[i]) * 1.55
    ax3.text(x3[i], y_lab, ratios_txt[i], ha="center",
             fontsize=6.0, color="#cc5500", weight="bold")

sci.save(fig, "fig08_deployment.png", w=7.4, h=3.0)
print("Saved fig08_deployment.png")
