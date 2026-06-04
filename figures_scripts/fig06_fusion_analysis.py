"""
Figure 6: Multimodal fusion analysis.

Data source: safety_data.py (EXP06: progressive_f1, fold_weights, architecture).
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
from sci_style import tnr_text
from safety_data import (
    EXP06_PROGRESSIVE_F1, EXP06_FOLD_WEIGHTS, EXP06_MEAN_WEIGHTS,
    EXP06_ARCHITECTURE,
)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.6, 2.7),
                                     gridspec_kw={"wspace": 0.55})

# ---- (a) Progressive modality contribution ----
modalities = [r["modality"] for r in EXP06_PROGRESSIVE_F1]
f1_prog    = [r["f1"]       for r in EXP06_PROGRESSIVE_F1]
deltas     = [r["delta"]    for r in EXP06_PROGRESSIVE_F1]

x = np.arange(len(modalities))
colors_prog = ["#bbbbbb", "#aaccdd", "#7fa9c8", "#ff7f0e"]
bars = ax1.bar(x, f1_prog, width=0.65, color=colors_prog,
               edgecolor="black", lw=0.5)
bars[-1].set_edgecolor("#cc5500")
bars[-1].set_linewidth(1.3)

ax1.set_xticks(x)
ax1.set_xticklabels(modalities, fontsize=7)
ax1.set_ylabel("F1 score")
ax1.set_ylim(0.85, 0.94)
ax1.set_title("(a) Modality contribution",
              weight="bold", fontsize=9)

for i in range(1, len(modalities)):
    tnr_text(ax1, x[i], f1_prog[i] + 0.005, f"+{deltas[i]:.3f}",
             ha="center", color="#cc5500", weight="bold")

# ---- (b) 5-fold CV weight stability ----
folds   = [r["fold"]    for r in EXP06_FOLD_WEIGHTS]
w_text  = [r["w_text"]  for r in EXP06_FOLD_WEIGHTS]
w_audio = [r["w_audio"] for r in EXP06_FOLD_WEIGHTS]
w_url   = [r["w_url"]   for r in EXP06_FOLD_WEIGHTS]
w_meta  = [r["w_meta"]  for r in EXP06_FOLD_WEIGHTS]

mw = EXP06_MEAN_WEIGHTS
ax2.plot(folds, w_text,  "o-", color="#1f77b4", lw=1.3, ms=5,
         label=f"$w_{{\\rm text}}$  ($\\mu$={mw['w_text']:.2f})")
ax2.plot(folds, w_audio, "s-", color="#ff7f0e", lw=1.3, ms=5,
         label=f"$w_{{\\rm audio}}$ ($\\mu$={mw['w_audio']:.2f})")
ax2.plot(folds, w_url,   "^-", color="#2ca02c", lw=1.3, ms=5,
         label=f"$w_{{\\rm url}}$  ($\\mu$={mw['w_url']:.2f})")
ax2.plot(folds, w_meta,  "D-", color="#9467bd", lw=1.3, ms=5,
         label=f"$w_{{\\rm meta}}$ ($\\mu$={mw['w_meta']:.2f})")

for mu, c in [(mw["w_text"], "#1f77b4"), (mw["w_audio"], "#ff7f0e"),
              (mw["w_url"], "#2ca02c"),  (mw["w_meta"], "#9467bd")]:
    ax2.axhline(mu, color=c, lw=0.4, ls=":", alpha=0.5)

ax2.set_xlabel("CV fold")
ax2.set_ylabel("L-BFGS fusion weight")
ax2.set_xticks(folds)
ax2.set_ylim(0.05, 0.48)
ax2.set_title("(b) 5-fold CV stability",
              weight="bold", fontsize=9)
ax2.legend(loc="center right", fontsize=6.3, ncol=1)

# ---- (c) Architecture comparison ----
for a in EXP06_ARCHITECTURE:
    name   = a["arch"]
    f1     = a["f1"]
    lat    = a["latency_ms"]
    params = a["params"]
    is_ours = name.startswith("sigmoid")
    color  = "#ff7f0e" if is_ours else (
             "#7f7f7f" if "softmax" in name else
             "#1f77b4" if "2L" in name else "#9467bd")
    ms = np.log10(max(params, 5)) * 4 + 4
    ax3.scatter([lat], [f1], s=ms * 15, color=color,
                edgecolor="black", lw=0.6, alpha=0.85, zorder=3)
    ax3.annotate(name, xy=(lat, f1), xytext=(lat, f1 + 0.003),
                 ha="center", fontsize=6.5, weight="bold")

# Pareto front line
xs = [a["latency_ms"] for a in EXP06_ARCHITECTURE if a["arch"] != "softmax"]
ys = [a["f1"]         for a in EXP06_ARCHITECTURE if a["arch"] != "softmax"]
ax3.plot(xs, ys, "--", color="#888", lw=0.6, alpha=0.7)

ax3.set_xlabel("Latency (ms, log scale)")
ax3.set_ylabel("F1 score")
ax3.set_xscale("log")
ax3.set_xlim(0.3, 30)
ax3.set_ylim(0.905, 0.932)
ax3.set_title("(c) Arch. trade-off",
              weight="bold", fontsize=9)
ax3.text(8, 0.907, "circle $\\propto$ log(#params)",
         fontsize=6, color="#666", style="italic", ha="center")

sci.save(fig, "fig06_fusion_analysis.png", w=7.6, h=2.7)
print("Saved fig06_fusion_analysis.png")
