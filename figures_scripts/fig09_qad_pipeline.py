"""Figure 9: QAD training pipeline with OV-Freeze schedule."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import sci_style as sci

fig = plt.figure(figsize=(7.16, 3.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

# ---- (a) Schematic of training pipeline ----
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 8)
ax1.axis("off")
ax1.set_title("(a) QAD training pipeline", weight="bold", fontsize=9.5,
              loc="left", x=0.0)

# Input
inp = FancyBboxPatch((4, 6.5), 2.0, 0.9,
                     boxstyle="round,pad=0.05,rounding_size=0.1",
                     fc="#FFFFFF", ec="#222", lw=0.9)
ax1.add_patch(inp)
ax1.text(5, 6.95, "Input  $x$", ha="center", va="center", fontsize=9)

# Teacher
teacher = FancyBboxPatch((0.4, 4.0), 3.5, 1.7,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         fc="#E8F1F8", ec="#1f77b4", lw=1.3)
ax1.add_patch(teacher)
ax1.text(2.15, 5.30, "BF16 Teacher", ha="center", weight="bold", fontsize=8.5,
         color="#1f4060")
ax1.text(2.15, 4.85, "Qwen2.5-0.5B-Instruct", ha="center", fontsize=7,
         color="#1f4060")
ax1.text(2.15, 4.40, "(weights frozen)", ha="center", fontsize=6.5,
         style="italic", color="#1f4060")

# Student
student = FancyBboxPatch((6.1, 4.0), 3.5, 1.7,
                         boxstyle="round,pad=0.05,rounding_size=0.12",
                         fc="#FFF4E6", ec="#ff7f0e", lw=1.3)
ax1.add_patch(student)
ax1.text(7.85, 5.30, "NVFP4 Student", ha="center", weight="bold",
         fontsize=8.5, color="#8a4a00")
ax1.text(7.85, 4.85, "Q4_K_M variant for edge", ha="center", fontsize=7,
         color="#8a4a00")
ax1.text(7.85, 4.40, "(weights trainable)", ha="center", fontsize=6.5,
         style="italic", color="#8a4a00")

# OV-Freeze (within student, last 30%)
ovf = FancyBboxPatch((6.45, 2.5), 2.8, 0.9,
                     boxstyle="round,pad=0.05,rounding_size=0.1",
                     fc="#FFE9D6", ec="#cc5500", lw=1.0, ls="--")
ax1.add_patch(ovf)
ax1.text(7.85, 2.95, "OV-Freeze on q,k,v,o", ha="center", fontsize=7.5,
         color="#7a3000")
ax1.text(7.85, 2.65, "(last 30% of training steps)",
         ha="center", fontsize=6.3, style="italic", color="#7a3000")

# Logits + KL
kl_box = FancyBboxPatch((3.2, 1.0), 3.6, 1.0,
                        boxstyle="round,pad=0.05,rounding_size=0.1",
                        fc="#FFFFFF", ec="#2ca02c", lw=1.2)
ax1.add_patch(kl_box)
ax1.text(5.0, 1.65, r"$\mathcal{L}_{\rm QAD}=D_{\rm KL}(\,p_T(y|x)\,\|\,p_S(y|x)\,)$",
         ha="center", fontsize=8, color="#1e5a23")
ax1.text(5.0, 1.20, "Pure KL @ T = 1.0", ha="center", fontsize=6.8,
         style="italic", color="#1e5a23")

# Arrows
ax1.annotate("", xy=(2.15, 4.0), xytext=(4.5, 6.5),
             arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f77b4"))
ax1.annotate("", xy=(7.85, 4.0), xytext=(5.5, 6.5),
             arrowprops=dict(arrowstyle="->", lw=1.0, color="#ff7f0e"))

ax1.annotate("", xy=(4.0, 1.5), xytext=(2.15, 4.0),
             arrowprops=dict(arrowstyle="->", lw=1.0, color="#1f77b4"))
ax1.text(2.7, 2.7, "$p_T$", fontsize=7.5, color="#1f77b4", weight="bold")

ax1.annotate("", xy=(6.0, 1.5), xytext=(7.85, 4.0),
             arrowprops=dict(arrowstyle="->", lw=1.0, color="#ff7f0e"))
ax1.text(7.2, 2.7, "$p_S$", fontsize=7.5, color="#ff7f0e", weight="bold")

# Backprop to student only
ax1.annotate("", xy=(8.5, 4.0), xytext=(6.5, 2.0),
             arrowprops=dict(arrowstyle="->", lw=1.2, color="#cc5500",
                             connectionstyle="arc3,rad=0.3"))
ax1.text(8.4, 3.1, "$\\nabla\\theta_S$", fontsize=8,
         color="#cc5500", weight="bold")

# ---- (b) Training schedule visualization ----
ax2.set_title("(b) Training schedule & key milestones",
              weight="bold", fontsize=9.5, loc="left", x=0.0)
steps = np.arange(0, 2001)
lr = np.where(steps < 100, 1e-5 * steps / 100,
              1e-5 * 0.5 * (1 + np.cos(np.pi * (steps - 100) / 1900)))
ax2.plot(steps, lr * 1e5, color="#1f77b4", lw=1.5, label="LR (×$10^{-5}$)")
ax2.set_xlabel("Training step")
ax2.set_ylabel("Learning rate (×$10^{-5}$)", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
ax2.set_xlim(0, 2000)
ax2.set_ylim(0, 1.15)

# Shade OVF region (last 30% = steps 1400-2000)
ax2.axvspan(1400, 2000, color="#ffe9d6", alpha=0.6, zorder=0)
ax2.text(1700, 0.95, "OV-Freeze ON\n(last 30% steps)",
         ha="center", fontsize=7, color="#7a3000", weight="bold")

# Annotate phases
ax2.annotate("warm-up\n(100 steps)", xy=(50, 0.45), xytext=(250, 0.7),
             fontsize=6.5, arrowprops=dict(arrowstyle="->", lw=0.6),
             ha="left", color="#444")
ax2.annotate("cosine decay", xy=(700, 0.92), xytext=(700, 1.05),
             fontsize=6.8, ha="center", color="#1f4060")

# Total tokens annotation
ax2.text(1000, 0.05, "Total: 0.5 B tokens  (≈ 1.7% of original SFT corpus)",
         ha="center", fontsize=7, color="#444", style="italic")

sci.save(fig, "fig09_qad_pipeline.png", w=7.16, h=3.4)
print("Saved fig09_qad_pipeline.png")
