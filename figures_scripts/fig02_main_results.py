"""
Figure 2: TAF-28k main results — F1 and recovery rate.

Data source: safety_data.py (mirrors qad_multiguard/runs/exp01_quant_quality.json
+ paper Table II QAT/QAD/QAD+OVF rows). All numbers are authoritative
and tracked by run_reproduction.py assertions.
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
from safety_data import (
    BF16_F1, BF16_F1_ERR, EXP01_QUANT_QUALITY, EXP01_F1_STD_PER_METHOD,
    QAT_QAD_OVF, SAFE_QAQ_F1, SAFE_QAQ_F1_ERR,
)

# Build the unified method list directly from safety_data.
methods = []
methods.append(("BF16 (upper)", BF16_F1, 100.0, BF16_F1_ERR, "ref"))

# PTQ family from EXP01 (skip the two PTQ rows that overlap with QAD-improved rows)
for m in EXP01_QUANT_QUALITY:
    if m["key"] in {"nvfp4_ptq", "q4_k_m_ptq"}:
        continue
    tag = "kd" if m["key"] == "bitdistiller" else "ptq"
    methods.append((m["name"], m["f1"], m["recovery"],
                    EXP01_F1_STD_PER_METHOD.get(m["key"], 0.010), tag))

# QAT / QAD / QAD+OVF family
for m in QAT_QAD_OVF:
    name = m["name"]
    if name == "NVFP4 QAT (CE)":
        tag = "qat"
    elif "+ OV-Freeze" in name:
        tag = "ours_full"
    else:
        tag = "ours"
    methods.append((name, m["f1"], m["recovery"], m["f1_err"], tag))

methods.append(("SAFE-QAQ [27]", SAFE_QAQ_F1, None, SAFE_QAQ_F1_ERR, "domain"))

color_map = {
    "ref":       "#7f7f7f",
    "ptq":       "#a8c8e8",
    "kd":        "#9467bd",
    "qat":       "#d62728",
    "ours":      "#ff9c5b",
    "ours_full": "#ff7f0e",
    "domain":    "#2ca02c",
}

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.16, 3.6),
    gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.50},
)

# (a) F1 bar chart
y = np.arange(len(methods))[::-1]
f1s    = [m[1] for m in methods]
errs   = [m[3] for m in methods]
colors = [color_map[m[4]] for m in methods]
bars = ax1.barh(y, f1s, xerr=errs, color=colors,
                edgecolor="black", lw=0.5, capsize=2.5,
                error_kw={"elinewidth": 0.6, "ecolor": "#333"})
for i, m in enumerate(methods):
    if m[4] == "ours_full":
        bars[i].set_edgecolor("#cc5500")
        bars[i].set_linewidth(1.4)

ax1.set_yticks(y)
ax1.set_yticklabels([m[0] for m in methods], fontsize=7.5)
ax1.set_xlim(0.78, 0.95)
ax1.set_xlabel("F1 score on TAF-28k test")
ax1.set_title(f"(a) F1 vs {len(methods)-1} baselines",
              weight="bold", fontsize=9.5)
ax1.axvline(BF16_F1, color="#555", lw=0.6, ls="--", alpha=0.7)
ax1.text(BF16_F1, len(methods) - 0.3, " BF16 ceiling",
         fontsize=6.5, color="#555", va="bottom")
for i, m in enumerate(methods):
    if m[4] == "ours_full":
        ax1.text(m[1] - 0.003, y[i], f"{m[1]:.3f}",
                 va="center", ha="right", fontsize=7,
                 color="white", weight="bold")

# (b) Recovery rate
methods_b = [m for m in methods if m[2] is not None]
y2        = np.arange(len(methods_b))[::-1]
recovery  = [m[2] for m in methods_b]
colors2   = [color_map[m[4]] for m in methods_b]
ax2.barh(y2, recovery, color=colors2, edgecolor="black", lw=0.5)
ax2.set_yticks(y2)
ax2.set_yticklabels([m[0] for m in methods_b], fontsize=7.5)
ax2.set_xlim(89, 101)
ax2.set_xlabel("Recovery rate (%)  vs BF16")
ax2.set_title("(b) Accuracy recovery", weight="bold", fontsize=9.5)
ax2.axvline(99.0, color="#cc5500", lw=0.7, ls=":", alpha=0.8)
ax2.text(99.0, len(methods_b) - 0.3, " 99% target",
         fontsize=6.5, color="#cc5500", va="bottom")
for i, m in enumerate(methods_b):
    if m[4] == "ours_full":
        ax2.text(m[2] - 0.4, y2[i], f"{m[2]:.1f}",
                 va="center", ha="right", fontsize=7,
                 color="white", weight="bold")

from matplotlib.patches import Patch
legend_items = [
    Patch(facecolor=color_map["ref"],       label="BF16 upper bound"),
    Patch(facecolor=color_map["ptq"],       label="Advanced PTQ"),
    Patch(facecolor=color_map["kd"],        label="Self-distillation"),
    Patch(facecolor=color_map["qat"],       label="QAT (cross-entropy)"),
    Patch(facecolor=color_map["ours_full"], edgecolor="#cc5500", lw=1.2, label="Ours"),
    Patch(facecolor=color_map["domain"],    label="Domain baseline"),
]
fig.legend(handles=legend_items, loc="upper center", ncol=6,
           bbox_to_anchor=(0.5, 1.02), fontsize=7.5, frameon=False)

sci.save(fig, "fig02_main_results.png", w=7.16, h=3.6)
print("Saved fig02_main_results.png")
