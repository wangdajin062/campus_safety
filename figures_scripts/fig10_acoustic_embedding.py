"""Figure 10: 128-d non-invertible acoustic embedding F_v construction."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sci_style as sci

fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.6),
                          gridspec_kw={"wspace": 0.35,
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
ax.set_title("(a) Original log-mel spectrogram\n(full temporal information)",
             fontsize=8.5, weight="bold")
ax.text(100, 70, "Length: $T \\sim 300$ frames", ha="center",
        fontsize=6.5, color="#444", style="italic")

# (b) Time-averaged 64-d MFCC
ax = axes[1]
mfcc_avg = mel.mean(axis=1)
ax.bar(np.arange(64), mfcc_avg, color="#1f77b4", edgecolor="none", width=0.95)
ax.set_xlabel("MFCC coefficient")
ax.set_ylabel("Time-averaged log-magnitude")
ax.set_title("(b) Time-averaged 64-d MFCC\n(phoneme order destroyed)",
             fontsize=8.5, weight="bold")
ax.set_xlim(-0.5, 63.5)
ax.set_ylim(0, mfcc_avg.max() * 1.2)
# Annotation
ax.text(32, mfcc_avg.max() * 1.05,
        "non-invertible step 1: $\\sum_t \\to \\bar{f}_{\\rm mfcc}$",
        ha="center", fontsize=6.7, color="#7a3000", style="italic")

# (c) Final 128-d F_v = [f_mfcc ; W h_w]
ax = axes[2]
# Generate synthetic Whisper projection
W_proj = rng.normal(0, 1.0 / np.sqrt(384), size=(64, 384))
h_w = rng.normal(0, 1, 384)
proj = W_proj @ h_w
F_v = np.concatenate([mfcc_avg, proj])
# Normalize for display
F_v_norm = (F_v - F_v.min()) / (F_v.max() - F_v.min())

cols = ["#1f77b4"] * 64 + ["#ff7f0e"] * 64
ax.bar(np.arange(128), F_v_norm, color=cols, edgecolor="none", width=0.95)
ax.set_xlabel("$F_v$ dimension")
ax.set_ylabel("Normalised value")
ax.set_title("(c) Final 128-d $F_v$ for transmission\n(MFCC + Whisper projection)",
             fontsize=8.5, weight="bold")
ax.set_xlim(-0.5, 127.5)
ax.axvline(63.5, color="#444", lw=0.8, ls="--", alpha=0.6)
ax.text(32, 1.10, "$\\bar{f}_{\\rm mfcc}\\in\\mathbb{R}^{64}$", ha="center",
        fontsize=7, color="#1f4060", weight="bold")
ax.text(96, 1.10, "$W_{\\rm proj}\\,\\bar{h}_w\\in\\mathbb{R}^{64}$", ha="center",
        fontsize=7, color="#8a4a00", weight="bold")
ax.set_ylim(0, 1.25)

# Bottom caption
fig.text(0.5, -0.02,
         "Both steps (time-averaging + CLS pooling) drop $O(T)$ bits of frame-level information; "
         "WER $\\geq$ 0.95 in white-box + black-box GLO attacks (Table XIII)",
         ha="center", fontsize=6.8, style="italic", color="#444")

sci.save(fig, "fig10_acoustic_embedding.png", w=7.16, h=2.6)
print("Saved fig10_acoustic_embedding.png")
