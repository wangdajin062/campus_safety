"""Figure 10: 128-d non-invertible acoustic embedding F_v construction."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sci_style as sci

fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8),
                          gridspec_kw={"wspace": 0.50,
                                       "width_ratios": [1, 1, 1.1]})

# Generate a synthetic mel-spectrogram-like image
rng = np.random.RandomState(7)
n_time, n_mel = 200, 64
# Make patterns that look like speech (formant bands)
t = np.arange(n_time)
m = np.arange(n_mel)
T, M = np.meshgrid(t, m, indexing="xy")
# Sinusoidal "voicing" + decaying spectrum
mel = (np.sin(T * 0.07) * np.exp(-((M - 10) ** 2) / 60) +
       np.cos(T * 0.05 + 1.2) * np.exp(-((M - 25) ** 2) / 90) * 0.7 +
       np.sin(T * 0.10 + 2.0) * np.exp(-((M - 45) ** 2) / 200) * 0.5)
mel += rng.normal(0, 0.10, size=mel.shape)
mel = mel - mel.min()
mel = mel / mel.max()

# (a) Original mel spectrogram
ax = axes[0]
im = ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
ax.set_xlabel("Time (frames)")
ax.set_ylabel("Mel bin")
ax.set_title("(a) Original log-mel\nspectrogram",
             fontsize=8.5, weight="bold")
# "Length: T~300" annotation placed in top-right corner of axes coords
ax.text(0.98, 0.96, "$T \\sim 300$ frames",
        ha="right", va="top", transform=ax.transAxes,
        fontsize=6.5, color="white", style="italic",
        bbox=dict(facecolor="black", alpha=0.5, pad=2, edgecolor="none"))

# (b) Time-averaged 64-d MFCC
ax = axes[1]
mfcc_avg = mel.mean(axis=1)
ax.bar(np.arange(64), mfcc_avg, color="#1f77b4", edgecolor="none", width=0.95)
ax.set_xlabel("MFCC coefficient")
ax.set_ylabel("Time-averaged log-magnitude")
ax.set_title("(b) Time-averaged\n64-d MFCC",
             fontsize=8.5, weight="bold")
ax.set_xlim(-0.5, 63.5)
ax.set_ylim(0, mfcc_avg.max() * 1.30)
# Annotation moved higher to avoid bar overlap
ax.text(32, mfcc_avg.max() * 1.18,
        "non-invertible step 1: $\\sum_t \\to \\bar{f}_{\\rm mfcc}$",
        ha="center", fontsize=6.5, color="#7a3000", style="italic")

# (c) Final 128-d F_v = [f_mfcc ; W h_w]
ax = axes[2]
W_proj = rng.normal(0, 1.0 / np.sqrt(384), size=(64, 384))
h_w = rng.normal(0, 1, 384)
proj = W_proj @ h_w
F_v = np.concatenate([mfcc_avg, proj])
F_v_norm = (F_v - F_v.min()) / (F_v.max() - F_v.min())

cols = ["#1f77b4"] * 64 + ["#ff7f0e"] * 64
ax.bar(np.arange(128), F_v_norm, color=cols, edgecolor="none", width=0.95)
ax.set_xlabel("$F_v$ dimension")
ax.set_ylabel("Normalised value")
ax.set_title("(c) Final 128-d $F_v$\nfor transmission",
             fontsize=8.5, weight="bold")
ax.set_xlim(-0.5, 127.5)
ax.axvline(63.5, color="#444", lw=0.8, ls="--", alpha=0.6)
# Move sub-labels higher to clear bars
ax.text(32, 1.18, "$\\bar{f}_{\\rm mfcc}\\in\\mathbb{R}^{64}$", ha="center",
        fontsize=7, color="#1f4060", weight="bold")
ax.text(96, 1.18, "$W_{\\rm proj}\\,\\bar{h}_w\\in\\mathbb{R}^{64}$", ha="center",
        fontsize=7, color="#8a4a00", weight="bold")
ax.set_ylim(0, 1.30)

# Bottom caption — move further down to clear x-axis labels
fig.text(0.5, -0.07,
         "Both steps (time-averaging + CLS pooling) drop $O(T)$ bits of frame-level information; "
         "WER $\\geq$ 0.95 in white-box + black-box GLO attacks",
         ha="center", fontsize=6.5, style="italic", color="#090000FF")

sci.save(fig, "fig10_acoustic_embedding.png", w=7.6, h=2.9)
print("Saved fig10_acoustic_embedding.png")
