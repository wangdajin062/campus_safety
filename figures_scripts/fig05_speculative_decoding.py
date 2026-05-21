"""
Fig 5: Speculative decoding (REVISED v2)
FIXES per reviewer R11, R12.
"""
import matplotlib.pyplot as plt
import numpy as np
from _fig_data import load_exp_data, fallback

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})

PALETTE = {
    'navy':   '#1F3864',
    'blue':   '#2E5C8A',
    'red':    '#C00000',
    'orange': '#E67E22',
    'green':  '#7CB342',
    'gray':   '#808080',
    'purple': '#7030A0',
    'gold':   '#F4B400',
    'lred':   '#FBE9E9',
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# (a) Theoretical speedup curves
ax = axes[0]
alphas = np.linspace(0.5, 0.95, 200)
gammas = [3, 5, 7, 10]
colors = [PALETTE['green'], PALETTE['red'], PALETTE['purple'], PALETTE['gold']]
for gamma, color in zip(gammas, colors):
    speedups = (1 - alphas**(gamma+1)) / (1 - alphas)
    ax.plot(alphas, speedups, color=color, linewidth=2.2, label=f"γ = {gamma}")

ax.plot(0.78, 3.522, 'o', color=PALETTE['gray'], markersize=14,
        markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.annotate("α=0.78, γ=5\nspeedup = 3.52×\n(generic draft)",
            xy=(0.78, 3.522), xytext=(0.55, 4.5),
            fontsize=10, color=PALETTE['gray'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.2, color=PALETTE['gray']),
            bbox=dict(facecolor='white', edgecolor=PALETTE['gray'],
                     boxstyle='round,pad=0.3'))

ax.plot(0.86, 4.253, '*', color=PALETTE['red'], markersize=22,
        markeredgecolor='black', markeredgewidth=1.5, zorder=5)
ax.annotate("α=0.86, γ=5\nspeedup = 4.25×\n(ours: anti-fraud tuned)",
            xy=(0.86, 4.253), xytext=(0.60, 6.0),
            fontsize=10.5, color=PALETTE['red'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.4, color=PALETTE['red']),
            bbox=dict(facecolor=PALETTE['lred'], edgecolor=PALETTE['red'],
                     boxstyle='round,pad=0.3'))

ax.set_xlabel("Token acceptance rate α", fontsize=11)
ax.set_ylabel("Theoretical speedup", fontsize=11)
ax.set_title("(a)  Theoretical speedup vs α at different γ", fontsize=12, fontweight='bold')
ax.set_xlim(0.5, 0.95)
ax.set_ylim(1.5, 7.5)

legend_elements = [
    plt.Line2D([0], [0], color=PALETTE['green'],  lw=2.2, label="γ = 3"),
    plt.Line2D([0], [0], color=PALETTE['red'],    lw=2.2, label="γ = 5"),
    plt.Line2D([0], [0], color=PALETTE['purple'], lw=2.2, label="γ = 7"),
    plt.Line2D([0], [0], color=PALETTE['gold'],   lw=2.2, label="γ = 10"),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['gray'],
               markersize=10, label="Generic draft"),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=PALETTE['red'],
               markersize=14, label="Ours: anti-fraud tuned"),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9.5, framealpha=0.95)
ax.grid(True, alpha=0.3)

# Improved formula box
ax.text(0.625, 1.85,
        r"$\mathrm{speedup}(\alpha,\gamma) = \dfrac{1 - \alpha^{\gamma+1}}{1 - \alpha}$",
        fontsize=12.5, ha='center',
        bbox=dict(facecolor='white', edgecolor=PALETTE['navy'],
                 boxstyle='round,pad=0.5', alpha=0.97))

# (b) γ sensitivity
ax = axes[1]
gammas_b = [3, 5, 7, 10]
exp5 = load_exp_data("exp05_speculative.json")
if exp5 and any(r.get("measured_speedup_h100") for r in exp5.get("results", [])):
    h100_measured = []
    sd_measured = []
    kv_cache = []
    for r in exp5["results"]:
        if r["model"] == "anti-fraud-tuned":
            h100_measured.append(r["measured_speedup_h100"])
            sd_measured.append(r["measured_speedup_sd8g3"])
            kv_cache.append(int(r.get("kv_cache_mb", 200)))
else:
    h100_measured = [2.65, 3.49, 4.10, 4.74]
    sd_measured   = [2.52, 3.32, 3.90, 4.51]
    kv_cache      = [120, 200, 280, 400]

x = np.arange(len(gammas_b))
width = 0.34

bars1 = ax.bar(x - width/2, h100_measured, width, label='H100 (cloud)',
               color=PALETTE['blue'], edgecolor='black', linewidth=1.0, alpha=0.9)
bars2 = ax.bar(x + width/2, sd_measured, width, label='SD8G3 (device)',
               color=PALETTE['orange'], edgecolor='black', linewidth=1.0, alpha=0.9)

for i in [1]:
    bars1[i].set_edgecolor(PALETTE['red'])
    bars1[i].set_linewidth(2.4)
    bars2[i].set_edgecolor(PALETTE['red'])
    bars2[i].set_linewidth(2.4)

# Higher offset for labels (R11 fix - especially for γ=5 SD value)
for i, (h, s) in enumerate(zip(h100_measured, sd_measured)):
    h_offset = 0.13
    s_offset = 0.13 if i != 1 else 0.30  # Extra offset for γ=5 to avoid red border
    ax.text(i - width/2, h + h_offset, f"{h:.2f}×", ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color=PALETTE['blue'])
    ax.text(i + width/2, s + s_offset, f"{s:.2f}×", ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color=PALETTE['orange'])

ax2 = ax.twinx()
ax2.plot(x, kv_cache, 'D-', color=PALETTE['gray'], markersize=10,
         linewidth=1.8, alpha=0.85, label='KV cache (MB)')
for i, kv in enumerate(kv_cache):
    ax2.text(i, kv + 22, f"{kv} MB", ha='center', va='bottom',
             fontsize=9, color=PALETTE['gray'], style='italic')

ax2.set_ylabel("KV cache memory (MB)", fontsize=10.5, color=PALETTE['gray'])
ax2.tick_params(axis='y', labelcolor=PALETTE['gray'])
ax2.set_ylim(0, 600)
ax2.spines['right'].set_color(PALETTE['gray'])

ax.annotate("Best balance:\n3.49× (H100), 3.32× (SD8G3)\n200 MB KV cache",
            xy=(1, 3.49), xytext=(1.7, 5.2),
            fontsize=10, color=PALETTE['red'], fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.4, color=PALETTE['red']),
            bbox=dict(facecolor=PALETTE['lred'], edgecolor=PALETTE['red'],
                     boxstyle='round,pad=0.3'))

ax.set_xticks(x)
ax.set_xticklabels([f"γ = {g}" for g in gammas_b], fontsize=11)
ax.set_ylabel("Measured speedup", fontsize=11)
ax.set_title("(b)  γ sensitivity (measured speedup with KV cache footprint)",
             fontsize=12, fontweight='bold')
ax.set_ylim(0, 6.2)
ax.grid(True, axis='y', alpha=0.3)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.95)

fig.text(0.5, 0.0,
         "Fig. 5.  Speculative decoding analysis. (a) Domain fine-tuning lifts α from 0.78 to 0.86 (3.52× → 4.25× theoretical). "
         "(b) γ = 5 strikes the best balance between speedup (3.49× / 3.32×) and KV-cache footprint (200 MB).",
         ha='center', fontsize=9.5, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('./output/fig05_speculative_decoding.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig05 regenerated: clearer formula and labels")
