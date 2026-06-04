"""
Figure 3: Loss function and teacher selection ablations.

Data source: safety_data.py (EXP03 loss ablation + EXP09 teacher).
"""
import numpy as np
import matplotlib.pyplot as plt
import sci_style as sci
from sci_style import tnr_text
from safety_data import BF16_F1, EXP03_LOSS_ABLATION, EXP09_TEACHER

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.0),
                                gridspec_kw={"wspace": 0.50})

# ---- (a) Loss function ablation ----
labels_map = {
    "pure_kl":       "Pure KL\n(ours)",
    "mse":           "MSE",
    "cross_entropy": "CE\n(= QAT)",
    "three_term":    "3-term\nhybrid",
    "kl_task_reg":   "KL +\ntask",
}
losses = [labels_map[r["loss"]] for r in EXP03_LOSS_ABLATION]
f1     = [r["f1"]   for r in EXP03_LOSS_ABLATION]
kl     = [r["kl"]   for r in EXP03_LOSS_ABLATION]
f1_err = [r["std"]  for r in EXP03_LOSS_ABLATION]

x = np.arange(len(losses))
w = 0.36

bars1 = ax1.bar(x - w/2, f1, w, yerr=f1_err, color="#1f77b4",
                edgecolor="black", lw=0.5, capsize=2.5,
                error_kw={"elinewidth": 0.6}, label="F1 ($\\uparrow$)")
bars1[0].set_color("#ff7f0e")
bars1[0].set_edgecolor("#cc5500")
bars1[0].set_linewidth(1.2)

ax1.set_xticks(x)
ax1.set_xticklabels(losses, fontsize=7.5)
ax1.set_ylabel("F1 score", color="#1f77b4")
ax1.set_ylim(0.80, 0.95)
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.set_title("a) Loss function ablation",
              weight="bold", fontsize=9.5)
ax1.axhline(BF16_F1, color="#555", lw=0.6, ls="--", alpha=0.6)
ax1.text(4.4, BF16_F1 + 0.002, "BF16", fontsize=6.5, color="#555")

ax1b = ax1.twinx()
ax1b.grid(False)
ax1b.bar(x + w/2, kl, w, color="#d62728", edgecolor="black",
         lw=0.5, label="KL to teacher ($\\downarrow$)", alpha=0.85)
ax1b.set_ylabel("KL divergence to BF16 teacher", color="#d62728")
ax1b.set_ylim(0, 0.36)
ax1b.tick_params(axis="y", labelcolor="#d62728")
ax1b.spines["right"].set_visible(True)
ax1b.spines["right"].set_color("#d62728")
ax1b.spines["top"].set_visible(False)
ax1b.spines["left"].set_visible(False)

ax1.annotate("Best F1\n& lowest KL",
             xy=(0, f1[0]), xytext=(0.3, 0.935),
             fontsize=6.8, color="#cc5500", ha="left", weight="bold",
             arrowprops=dict(arrowstyle="->", lw=0.6, color="#cc5500"))

# ---- (b) Teacher selection ----
teacher  = [r["teacher"]              for r in EXP09_TEACHER]
f1_fixed = [r["f1_fixed"]             for r in EXP09_TEACHER]
f1_conv  = [r["f1_conv"]              for r in EXP09_TEACHER]
tokens   = [r["tokens_to_converge_B"] for r in EXP09_TEACHER]

x2 = np.arange(len(teacher))
w2 = 0.36

b1 = ax2.bar(x2 - w2/2, f1_fixed, w2, color="#1f77b4",
             edgecolor="black", lw=0.5, label="Fixed 0.5B tokens")
b2 = ax2.bar(x2 + w2/2, f1_conv, w2, color="#2ca02c",
             edgecolor="black", lw=0.5, label="To convergence")
b1[0].set_color("#ff7f0e")
b1[0].set_edgecolor("#cc5500")
b1[0].set_linewidth(1.2)

ax2.set_xticks(x2)
ax2.set_xticklabels(teacher, fontsize=7.5)
ax2.set_ylabel("F1 score")
ax2.set_ylim(0.85, 0.95)
ax2.set_title("b) Teacher selection ",
              weight="bold", fontsize=9.5)
ax2.legend(loc="upper right", fontsize=7)
ax2.axhline(BF16_F1, color="#555", lw=0.6, ls="--", alpha=0.6)
ax2.text(3.0, BF16_F1 + 0.002, "BF16", fontsize=6.5, color="#555")

for xi, t in zip(x2, tokens):
    # Place token-count label ABOVE the green "convergence" bar (not under)
    tnr_text(ax2, xi + w2/2, 0.920, f"{t:.1f}B",
             ha="center", color="#2ca02c", style="italic", weight="bold")

sci.save(fig, "fig03_ablation_loss_teacher.png", w=7.6, h=3.0)
print("Saved fig03_ablation_loss_teacher.png")
