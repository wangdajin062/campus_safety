"""
Figure 5: Speculative decoding theoretical/measured speed-up.

Data source: safety_data.py (EXP05). Speedup curve in (a) is computed
analytically; the operating-point markers and (b) measured bars are
from runs/exp05_speculative.json verbatim.
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
from sci_style import tnr_text
from safety_data import EXP05_SPECULATIVE, speedup

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.0),
                                gridspec_kw={"wspace": 0.20})

# ---- (a) Theoretical speedup curve ----
alpha_grid = np.linspace(0.5, 0.95, 200)
for g, color, lw in [(3, "#7f7f7f", 1.0),
                      (5, "#1f77b4", 1.7),
                      (7, "#2ca02c", 1.0),
                      (10, "#9467bd", 1.0)]:
    ax1.plot(alpha_grid, [speedup(a, g) for a in alpha_grid],
             color=color, lw=lw, label=f"$\\gamma$ = {g}")

# Operating points (from JSON)
generic_g5 = next(r for r in EXP05_SPECULATIVE[0.78] if r["gamma"] == 5)
tuned_g5   = next(r for r in EXP05_SPECULATIVE[0.86] if r["gamma"] == 5)

ax1.plot(0.78, generic_g5["theor"], "o", color="#d62728", ms=8,
         markeredgecolor="black", lw=0.5, zorder=5)
ax1.plot(0.86, tuned_g5["theor"], "*", color="#ff7f0e", ms=15,
         markeredgecolor="#cc5500", lw=0.8, zorder=5)

ax1.annotate(f"generic\n($\\alpha=0.78$, {generic_g5['theor']:.2f}×)",
             xy=(0.78, generic_g5["theor"]),
             xytext=(0.53, 3.5), fontsize=7, 
                           arrowprops=dict(arrowstyle="->", lw=0.7, color="#a02020"))
ax1.annotate(f"anti-fraud-tuned\n($\\alpha=0.86$, {tuned_g5['theor']:.2f}×)",
             xy=(0.86, tuned_g5["theor"]),
             xytext=(0.57, 4.4), fontsize=7,
             arrowprops=dict(arrowstyle="->", lw=0.7, color="#cc5500"))

ax1.set_xlabel(r"Token acceptance rate $\alpha$", fontsize=8.5)
ax1.set_ylabel("Theoretical speed-up",fontsize=8.5)
ax1.set_xlim(0.5, 0.95)
ax1.set_ylim(1.5, 6.5)
ax1.legend(loc="upper left", fontsize=7.5)
ax1.set_title("a) Speed-up curve, Leviathan et al.",
              weight="bold", fontsize=9.5)

# ---- (b) γ sensitivity (measured) at α=0.86 ----
tuned   = EXP05_SPECULATIVE[0.86]
gammas  = [r["gamma"] for r in tuned]
theor   = [r["theor"] for r in tuned]
meas_h  = [r["h100"]  for r in tuned]
meas_s  = [r["sd8g3"] for r in tuned]
kv_mb   = [r["kv_mb"] for r in tuned]

x = np.arange(len(gammas))
w = 0.25
ax2.bar(x - w, theor, w, color="#a8c8e8", edgecolor="black", lw=0.5,
        label="Theoretical")
ax2.bar(x, meas_h, w, color="#1f77b4", edgecolor="black", lw=0.5,
        label="Measured (H100)")
ax2.bar(x + w, meas_s, w, color="#ff7f0e", edgecolor="black", lw=0.5,
        label="Measured (SD8G3)")

ax2.set_xticks(x)
ax2.set_xticklabels([f"$\\gamma$={g}" for g in gammas])
ax2.set_ylabel("Speed-up factor (×)", fontsize=7.5)
ax2.set_ylim(0, 6.5)
ax2.set_title("b) $\\gamma$ sensitivity at $\\alpha=0.86$" ,
              weight="bold", fontsize=9.5)
ax2.legend(loc="upper left", fontsize=7.5)

ax2b = ax2.twinx()
ax2b.grid(False)
ax2b.plot(x, kv_mb, "D--", color="#2ca02c", lw=1.0, ms=4,
          label="KV-cache (MB)")
ax2b.set_ylabel("KV-cache footprint (MB)", fontsize=7.5, color="#2ca02c")
ax2b.tick_params(axis="y", labelcolor="#2ca02c")
ax2b.set_ylim(0, 5)
ax2b.spines["right"].set_visible(True)
ax2b.spines["right"].set_color("#2ca02c")
ax2b.spines["top"].set_visible(False)

# Mark γ=5 as the Pareto choice
ax2.axvline(1, color="#cc5500", lw=0.5, ls=":", alpha=0.6)
# Annotation placed below the γ=5 bar cluster, in the empty space at bottom
ax2.annotate("$\\gamma$=5 (ours)\nPareto-optimal",
             xy=(1, 3.49), xytext=(1, 4.5),
             ha="center", fontsize=7.5, color="#cc5500", weight="bold",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="#cc5500",
                             connectionstyle="arc3,rad=-0.2"))

sci.save(fig, "fig05_speculative.png", w=7.5, h=3.0)
print("Saved fig05_speculative.png")
