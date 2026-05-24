"""
Figure 7: Privacy verification — white-box + black-box GLO attacks.

Data source: safety_data.py (EXP07_PRIVACY).
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
from safety_data import EXP07_PRIVACY

fig = plt.figure(figsize=(7.16, 3.1))
ax1 = fig.add_subplot(1, 2, 1, polar=True)
ax2 = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(wspace=0.45)

# ---- (a) Radar of reconstruction quality (normalised so higher = more privacy) ----
labels = ["WER\n(↑ better)", "PESQ$^{-1}$", "MOS$^{-1}$",
          "Speaker-ID$^{-1}$", "$I(x;F_v)^{-1}$"]

def norm_pesq_mos(v): return (5 - v) / 4
def norm_spk(v):      return 1 - max(0, v - 0.10) / 0.9

def to_vec(d):
    return [d["wer"], norm_pesq_mos(d["pesq"]), norm_pesq_mos(d["mos"]),
            norm_spk(d["spk_id"]), 1.0]   # MI = 0 by construction

wb = to_vec(EXP07_PRIVACY["white_box"])
bb = to_vec(EXP07_PRIVACY["black_box"])
rand = to_vec(EXP07_PRIVACY["random_ref"])

n = len(labels)
angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
wb   += wb[:1]
bb   += bb[:1]
rand += rand[:1]
angles += angles[:1]

# PIPL compliance threshold band
threshold = [0.85] * (n + 1)
ax1.fill(angles, threshold, color="#2ca02c", alpha=0.10)
ax1.plot(angles, threshold, color="#2ca02c", lw=0.6, ls="--", alpha=0.7)

ax1.plot(angles, wb, "o-", color="#1f77b4", lw=1.3, ms=4, label="White-box")
ax1.fill(angles, wb, color="#1f77b4", alpha=0.15)
ax1.plot(angles, bb, "s-", color="#d62728", lw=1.3, ms=4, label="Black-box")
ax1.fill(angles, bb, color="#d62728", alpha=0.15)
ax1.plot(angles, rand, "^:", color="#7f7f7f", lw=0.8, ms=3, label="Random baseline")

ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(labels, fontsize=6.5)
ax1.set_yticks([0.25, 0.5, 0.75, 1.0])
ax1.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=6.3)
ax1.set_ylim(0, 1.05)
ax1.set_title("(a) Privacy metrics (PIPL §23)",
              weight="bold", fontsize=9, pad=15)
ax1.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05), fontsize=6.5)

# ---- (b) Scatter of WER vs Spkr-ID ----
wb_raw = EXP07_PRIVACY["white_box"]
bb_raw = EXP07_PRIVACY["black_box"]
rd_raw = EXP07_PRIVACY["random_ref"]

ax2.axhspan(0.05, 0.15, color="#E8F5E9", alpha=0.5, label="Compliance region")
ax2.axvspan(0.90, 1.00, color="#E8F5E9", alpha=0.5)

ax2.scatter([wb_raw["wer"]], [wb_raw["spk_id"]], s=140, color="#1f77b4",
            edgecolor="black", lw=0.8, zorder=4, label="White-box GLO")
ax2.scatter([bb_raw["wer"]], [bb_raw["spk_id"]], s=140, marker="s",
            color="#d62728", edgecolor="black", lw=0.8, zorder=4,
            label="Black-box inversion")
ax2.scatter([rd_raw["wer"]], [rd_raw["spk_id"]], s=120, marker="^",
            color="#7f7f7f", edgecolor="black", lw=0.6, zorder=4,
            label="Random (10 spk)")

ax2.annotate("PIPL §23\ncompliant",
             xy=(0.95, 0.10), xytext=(0.91, 0.18),
             fontsize=7, ha="center", color="#1e5a23", weight="bold",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="#1e5a23"))

ax2.set_xlabel("Reconstruction WER (↑ = more privacy)")
ax2.set_ylabel("Speaker-ID accuracy (↓ = more privacy)")
ax2.set_xlim(0.85, 1.02)
ax2.set_ylim(0, 0.3)
ax2.axhline(0.1, color="#888", lw=0.4, ls=":", alpha=0.6)
ax2.text(0.86, 0.105, "random 10-class baseline (10%)",
         fontsize=6, color="#666", style="italic")
ax2.set_title("(b) WER vs Speaker-ID, $n=100$",
              weight="bold", fontsize=9)
ax2.legend(loc="upper right", fontsize=6.5)

sci.save(fig, "fig07_privacy_glo.png", w=7.16, h=3.1)
print("Saved fig07_privacy_glo.png")
