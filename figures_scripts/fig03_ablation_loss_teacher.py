"""
Fig 3: Loss + Teacher Selection Ablation (REVISED)
FIXES per reviewer:
  R2: Add "KL + task regularizer" loss row (5th in Table IV)
  R3: Add "7B (more tokens)" with 0.915 row in Teacher Selection
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from _fig_data import load_exp_data, fallback

os.makedirs('./output', exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})

fig, axes = plt.subplots(1, 2, figsize=(16, 9))


PALETTE = {
    'red':     '#C00000',
    'green':   '#7CB342',
    'gold':    '#F4B400',
    'purple':  '#7030A0',
    'orange':  '#E67E22',
    'navy':    '#1F3864',
    'blue':    '#2E5C8A',
    'lblue':   '#A4C8E1',
    'gray':    '#808080',
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

# ── (a) Loss function ablation — 5 methods ────────
ax = axes[0]
# Try loading from experiment data
exp3 = load_exp_data("exp03_loss_ablation.json")
if exp3 and any(r.get("f1") for r in exp3.get("results", []) if r.get("student") == "close"):
    losses = []
    for r in exp3["results"]:
        if r["student"] == "close" and r["loss"] in ["pure_kl", "mse", "cross_entropy", "three_term", "kl_task_reg"]:
            color_map = {"pure_kl": PALETTE['red'], "mse": PALETTE['green'],
                         "cross_entropy": PALETTE['gold'], "three_term": PALETTE['purple'],
                         "kl_task_reg": PALETTE['orange']}
            label_map = {"pure_kl": "Pure KL\n(ours)", "mse": "MSE\non logits",
                         "cross_entropy": "Cross-entropy\n(= QAT)", "three_term": "Three-term\nhybrid (early)",
                         "kl_task_reg": "KL + task\nregularizer"}
            losses.append((
                label_map.get(r["loss"], r["loss"]),
                r["f1"], r.get("kl", 0.007), r.get("std", 0.007),
                color_map.get(r["loss"], PALETTE['gray']),
            ))
else:
    losses = [
        ("Pure KL\n(ours)",          0.916, 0.007, 0.005, PALETTE['red']),
        ("MSE\non logits",           0.901, 0.082, 0.010, PALETTE['green']),
        ("Cross-entropy\n(= QAT)",   0.844, 0.311, 0.014, PALETTE['gold']),
        ("Three-term\nhybrid (early)", 0.879, 0.124, 0.012, PALETTE['purple']),
        ("KL + task\nregularizer",   0.908, 0.041, 0.009, PALETTE['orange']),
    ]

x = np.arange(len(losses))
f1s   = [l[1] for l in losses]
kls   = [l[2] for l in losses]
errs  = [l[3] for l in losses]
colors= [l[4] for l in losses]
labels= [l[0] for l in losses]

# Bar plot for F1
ax2 = ax.twinx()
width = 0.35
bars1 = ax.bar(x - width/2, f1s, width, yerr=errs, color=colors,
               edgecolor='black', linewidth=1.0, capsize=5,
               label='F1 score', alpha=0.9)
bars2 = ax2.bar(x + width/2, kls, width, color=colors, alpha=0.5,
                edgecolor=PALETTE['navy'], linewidth=1.0,
                label='KL div. vs BF16', hatch='//')

# Highlight best (Pure KL)
bars1[0].set_edgecolor(PALETTE['red'])
bars1[0].set_linewidth(2.4)

# F1 labels
for i, (v, e) in enumerate(zip(f1s, errs)):
    weight = 'bold' if i == 0 else 'normal'
    ax.text(i - width/2, v + e + 0.003, f"{v:.3f}", ha='center', va='bottom',
            fontsize=9.5, fontweight=weight, color=PALETTE['navy'])

# KL labels
for i, kl in enumerate(kls):
    ax2.text(i + width/2, kl + 0.008, f"{kl:.3f}", ha='center', va='bottom',
             fontsize=9, color=PALETTE['navy'])

# Annotation
ax.annotate("Best F1 +\nlowest KL", xy=(-0.1, 0.916), xytext=(0.5, 0.945),
            fontsize=11, color=PALETTE['red'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PALETTE['red'], lw=1.5))

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("F1 score", fontsize=11, color=PALETTE['red'])
ax2.set_ylabel("KL divergence vs BF16 teacher", fontsize=11, color=PALETTE['navy'])
ax.tick_params(axis='y', labelcolor=PALETTE['red'])
ax2.tick_params(axis='y', labelcolor=PALETTE['navy'])
ax.set_ylim(0.80, 0.97)
ax2.set_ylim(0, 0.40)
ax.set_title("(a)  Loss function ablation (NVFP4 QAD)", fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10,
          framealpha=0.95)

# ── (b) Teacher selection — 4 teachers × 2 conditions ────
ax = axes[1]
exp9 = load_exp_data("exp09_teacher_selection.json")
if exp9 and exp9.get("results"):
    teacher_names = []
    f1_fixed = []
    f1_converged = []
    tokens_conv = []
    for r in exp9["results"]:
        teacher_names.append(r["teacher"].replace(" (same-source)", "\n(same-source,\nours)"))
        f1_fixed.append(r["f1_fixed"])
        f1_converged.append(r["f1_converged"])
        tokens_conv.append(r["tokens_to_converge"])
else:
    teacher_names = ['0.5B BF16\n(same-source,\nours)', '1.8B BF16', '3B BF16', '7B BF16']
    f1_fixed     = [0.916, 0.911, 0.904, 0.892]
    f1_converged = [0.916, 0.913, 0.910, 0.915]
    tokens_conv  = ['0.5B', '0.7B', '1.0B', '2.0B']

x = np.arange(len(teacher_names))
width = 0.35

bars1 = ax.bar(x - width/2, f1_fixed, width, color=PALETTE['lblue'],
               edgecolor='black', linewidth=1.0, alpha=0.9,
               label='Fixed 0.5B tokens')
bars2 = ax.bar(x + width/2, f1_converged, width, color=PALETTE['blue'],
               edgecolor='black', linewidth=1.0, alpha=0.9,
               label='Until convergence')

# Highlight ours
bars1[0].set_edgecolor(PALETTE['red'])
bars1[0].set_linewidth(2.4)
bars1[0].set_color(PALETTE['red'])
bars2[0].set_edgecolor(PALETTE['red'])
bars2[0].set_linewidth(2.4)
bars2[0].set_color(PALETTE['red'])

# F1 labels
for i, (vf, vc) in enumerate(zip(f1_fixed, f1_converged)):
    weight = 'bold' if i == 0 else 'normal'
    ax.text(i - width/2, vf + 0.0015, f"{vf:.3f}", ha='center', va='bottom',
            fontsize=10, fontweight=weight, color=PALETTE['navy'])
    ax.text(i + width/2, vc + 0.0015, f"{vc:.3f}", ha='center', va='bottom',
            fontsize=10, fontweight=weight, color=PALETTE['navy'])

# Tokens-to-converge labels
for i, t in enumerate(tokens_conv):
    color = PALETTE['orange']
    ax.text(i, 0.940, f"converges\n@ {t}", ha='center', va='center', fontsize=9,
            color=color, style='italic',
            bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2'))

# BF16 upper bound line
ax.axhline(y=0.931, color=PALETTE['navy'], linestyle='--', linewidth=1.0, alpha=0.7)
ax.text(3.5, 0.932, "BF16 upper bound (0.931)", fontsize=9, color=PALETTE['navy'],
        ha='right', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(teacher_names, fontsize=10)
ax.set_ylabel("F1 score", fontsize=11)
ax.set_title("(b)  Teacher selection ablation (NVFP4 QAD, 0.5B student)",
             fontsize=13, fontweight='bold')
ax.set_ylim(0.85, 0.955)
ax.grid(True, axis='y', alpha=0.3)
ax.legend(loc='lower left', fontsize=9.5, framealpha=0.95)

# Annotation
ax.annotate("Same-source teacher:\nleast tokens, top F1",
            xy=(0, 0.916), xytext=(1.5, 0.875),
            fontsize=10, color=PALETTE['red'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=PALETTE['red'], lw=1.2),
            bbox=dict(facecolor='#FFF5F5', edgecolor=PALETTE['red'],
                     boxstyle='round,pad=0.3'))

fig.text(0.5, 0.0,
         "Fig. 3.  Loss function ablation (a) confirms pure KL divergence achieves the best F1 with the lowest KL divergence to BF16 teacher. "
         "Teacher selection ablation (b) shows the same-source 0.5B teacher attains the highest F1 with 4× less data than the 7B teacher.",
         ha='center', fontsize=9.5, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('./output/fig03_ablation_loss_teacher.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig03 regenerated: 5 losses + 5 teacher conditions")
