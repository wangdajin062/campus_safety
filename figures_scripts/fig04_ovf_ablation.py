"""
Figure 4: OV-Freeze layer selection & training-step ratio ablation.

Data source: safety_data.py (EXP04 layer ablation + EXP10 step ratio).
Note: FFN-only and q,k,v,o+FFN rows are paper-extension points
(from_json=False); the other rows come directly from the JSON.
"""
import matplotlib.pyplot as plt
import sci_style as sci
from safety_data import EXP04_OVF_LAYER_ABLATION, EXP10_OVF_STEP_RATIO

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0),
                                gridspec_kw={"wspace": 0.35})

# ---- (a) Layer selection ablation ----
label_map = {
    "baseline":    "No OVF\n(baseline)",
    "FFN_only":    "FFN\nonly",
    "q_only":      "q\nonly",
    "q_v":         "q, v",
    "q_k_v":       "q, k, v",
    "q_k_v_o":     "q, k, v, o\n(ours)",
    "q_k_v_o+FFN": "q,k,v,o\n+ FFN",
}
configs = [label_map[r["config"]] for r in EXP04_OVF_LAYER_ABLATION]
f1      = [r["f1"]                 for r in EXP04_OVF_LAYER_ABLATION]
drift   = [r["drift_pct"]          for r in EXP04_OVF_LAYER_ABLATION]
from_json = [r["from_json"]        for r in EXP04_OVF_LAYER_ABLATION]

import numpy as np
x = np.arange(len(configs))

color1, color2 = "#1f77b4", "#d62728"
bars = ax1.bar(x, f1, color=color1, edgecolor="black", lw=0.5,
               alpha=0.85, label="F1 (↑)")

# Highlight the "ours" config (q_k_v_o)
ours_idx = next(i for i, r in enumerate(EXP04_OVF_LAYER_ABLATION)
                if r["config"] == "q_k_v_o")
bars[ours_idx].set_color("#ff7f0e")
bars[ours_idx].set_edgecolor("#cc5500")
bars[ours_idx].set_linewidth(1.3)

# Hatch the paper-extension bars
for i, fj in enumerate(from_json):
    if not fj:
        bars[i].set_hatch("///")
        bars[i].set_alpha(0.7)

ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=6.8)
ax1.set_ylabel("F1 score", color=color1)
ax1.set_ylim(0.910, 0.928)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.set_title("(a) Layer-selection ablation (Table VI)",
              weight="bold", fontsize=9.5)

ax1b = ax1.twinx()
ax1b.grid(False)
ax1b.plot(x, drift, "o-", color=color2, lw=1.5, ms=5, label="Var drift (↓)")
ax1b.set_ylabel("Output variance drift (%)", color=color2)
ax1b.set_ylim(0, 22)
ax1b.tick_params(axis="y", labelcolor=color2)
ax1b.spines["right"].set_visible(True)
ax1b.spines["right"].set_color(color2)
ax1b.spines["top"].set_visible(False)
ax1b.spines["left"].set_visible(False)

ax1.annotate("Best F1 +\nlowest drift",
             xy=(ours_idx, f1[ours_idx]), xytext=(2.3, 0.9255),
             fontsize=6.5, color="#cc5500", ha="center", weight="bold",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="#cc5500"))

# Footnote: explain hatched bars
ax1.text(0, 0.911, "hatched: paper extension; solid: from runs/exp04.json",
         fontsize=5.5, color="#666", style="italic")

# ---- (b) Step-ratio ablation ----
ratios = [r["ratio_pct"] for r in EXP10_OVF_STEP_RATIO]
f1_r   = [r["f1"]        for r in EXP10_OVF_STEP_RATIO]
ppl_r  = [r["ppl"]       for r in EXP10_OVF_STEP_RATIO]

ax2.plot(ratios, f1_r, "o-", color="#1f77b4", lw=1.5, ms=6, label="F1 (↑)")
ax2.plot(30, 0.923, "o", color="#ff7f0e", ms=11,
         markeredgecolor="#cc5500", lw=1.5, zorder=5)
ax2.set_xlabel("OV-Freeze training-step ratio (%)")
ax2.set_ylabel("F1 score", color="#1f77b4")
ax2.set_ylim(0.913, 0.926)
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_title("(b) Step-ratio ablation (Table VII)",
              weight="bold", fontsize=9.5)

ax2b = ax2.twinx()
ax2b.grid(False)
ax2b.plot(ratios, ppl_r, "s--", color="#d62728", lw=1.5, ms=5, label="PPL (↓)")
ax2b.set_ylabel("Perplexity", color="#d62728")
ax2b.tick_params(axis="y", labelcolor="#d62728")
ax2b.set_ylim(8.58, 8.78)
ax2b.spines["right"].set_visible(True)
ax2b.spines["right"].set_color("#d62728")
ax2b.spines["top"].set_visible(False)

ax2.axvspan(45, 50.5, color="#ffcccc", alpha=0.4, zorder=0)
ax2.text(47.5, 0.914, "gradient\noscillation", fontsize=6.3,
         ha="center", color="#a02020", style="italic")
ax2.axvline(30, color="#cc5500", lw=0.5, ls=":", alpha=0.7)
ax2.text(30, 0.9248, "ours: 30%", fontsize=6.5, color="#cc5500",
         ha="center", weight="bold")

sci.save(fig, "fig04_ovf_ablation.png", w=7.16, h=3.0)
print("Saved fig04_ovf_ablation.png")
