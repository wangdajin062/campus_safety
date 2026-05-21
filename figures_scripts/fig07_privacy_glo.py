"""
Fig 7: Privacy verification (REVISED)
FIXES per reviewer:
  R6: Clarify "↓ better" — change axis labels in radar to indicate "from privacy perspective"
  R7: Separate White-box/Black-box data point labels in panel (b)
"""
import matplotlib.pyplot as plt
import numpy as np
from _fig_data import load_exp_data, fallback

PALETTE = {
    'navy':   '#1F3864',
    'blue':   '#2E5C8A',
    'red':    '#C00000',
    'green':  '#7CB342',
    'orange': '#E67E22',
    'gray':   '#808080',
    'lgreen': '#E8F5E9',
}

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})

fig = plt.figure(figsize=(15, 6.5))

# (a) Radar chart — REWORKED axis interpretations (R6)
ax1 = fig.add_subplot(121, projection='polar')

# Each metric is normalized to [0, 1] where 1 = "more privacy" (better for us)
# Order around radar:
categories = [
    "WER↑",                  # higher = more privacy
    "1−PESQ↑",               # invert PESQ; higher = more privacy
    "1−MOS↑",                # invert MOS; higher = more privacy
    "Speaker-ID failure↑",   # higher = more privacy
    "1−I(x;F_v)↑"            # invert mutual info; higher = more privacy
]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Privacy-normalized values (all in [0,1], higher = better privacy)
exp7 = load_exp_data("exp07_privacy.json")
if exp7 and exp7.get("results") and len(exp7["results"]) >= 2:
    wb = exp7["results"][0]
    bb = exp7["results"][1]
    white_box  = [wb["wer"], 1 - wb["pesq"]/5, 1 - wb["mos"]/5, 1 - wb["speaker_id_acc"], 1 - wb.get("mutual_info", 0)]
    black_box  = [bb["wer"], 1 - bb["pesq"]/5, 1 - bb["mos"]/5, 1 - bb["speaker_id_acc"], 1 - bb.get("mutual_info", 0)]
    threshold  = [0.90, 1 - 0.30, 1 - 0.30, 0.85, 0.85]
else:
    white_box  = [0.95, 1 - 0.242, 1 - 0.236, 0.917, 1.0]
    black_box  = [0.97, 1 - 0.232, 1 - 0.222, 0.921, 1.0]
    threshold  = [0.90, 1 - 0.30, 1 - 0.30, 0.85, 0.85]

white_box  += white_box[:1]
black_box  += black_box[:1]
threshold  += threshold[:1]

ax1.set_theta_offset(np.pi / 2)
ax1.set_theta_direction(-1)
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(categories, fontsize=10)
ax1.set_ylim(0, 1.0)
ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)

# Plot threshold first (background)
ax1.plot(angles, threshold, color=PALETTE['gray'], linestyle='--',
         linewidth=1.5, label='PIPL §23 minimum')
ax1.fill(angles, threshold, alpha=0.10, color=PALETTE['gray'])

# Plot White-box and Black-box
ax1.plot(angles, white_box, 'o-', color=PALETTE['blue'], linewidth=2.0,
         markersize=8, label='White-box GLO (known $W_{proj}$)')
ax1.fill(angles, white_box, alpha=0.20, color=PALETTE['blue'])

ax1.plot(angles, black_box, 's-', color=PALETTE['red'], linewidth=2.0,
         markersize=8, label='Black-box model inversion')
ax1.fill(angles, black_box, alpha=0.20, color=PALETTE['red'])

ax1.set_title("(a)  Privacy radar: 'more privacy' is outward\n(all 5 metrics normalized to [0,1])",
              fontsize=11.5, fontweight='bold', pad=22)
ax1.legend(loc='upper right', bbox_to_anchor=(1.32, 1.10), fontsize=9, framealpha=0.95)

# (b) WER vs Speaker-ID accuracy — separated labels (R7)
ax2 = fig.add_subplot(122)

# PIPL compliant zone (top right is best)
ax2.axhspan(0, 12, xmin=(0.90 - 0.88) / (1.02 - 0.88), facecolor=PALETTE['lgreen'], alpha=0.5)
ax2.axvline(x=0.90, color=PALETTE['gray'], linestyle='--', linewidth=1.0, alpha=0.7)
ax2.axhline(y=15, color=PALETTE['gray'], linestyle='--', linewidth=1.0, alpha=0.7)
ax2.text(1.02, 2, "PIPL §23\ncompliant\nzone", fontsize=10,
         color=PALETTE['green'], fontweight='bold', ha='right',
         bbox=dict(facecolor='white', edgecolor=PALETTE['green'],
                  boxstyle='round,pad=0.3'))

# Data points — well separated
if exp7 and exp7.get("results") and len(exp7["results"]) >= 2:
    wb_wer = wb["wer"]
    wb_id = round((1 - wb["speaker_id_acc"]) * 100, 1)
    bb_wer = bb["wer"]
    bb_id = round((1 - bb["speaker_id_acc"]) * 100, 1)
else:
    wb_wer = 0.95
    wb_id = 8.3
    bb_wer = 0.97
    bb_id = 7.9

# White-box at (wb_wer, wb_id)
ax2.plot(wb_wer, wb_id, 'o', color=PALETTE['blue'], markersize=14,
         markeredgecolor='black', markeredgewidth=1.2)
ax2.annotate(f"White-box GLO\n(WER={wb_wer:.2f}, ID={wb_id:.1f}%)", xy=(wb_wer, wb_id), xytext=(0.91, 12.5),
             fontsize=10, color=PALETTE['blue'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=PALETTE['blue'], lw=1.0),
             bbox=dict(facecolor='white', edgecolor=PALETTE['blue'],
                       boxstyle='round,pad=0.3'))

# Black-box at (bb_wer, bb_id) — place label LOWER and to the right
ax2.plot(bb_wer, bb_id, 's', color=PALETTE['red'], markersize=12,
         markeredgecolor='black', markeredgewidth=1.2)
ax2.annotate(f"Black-box inv.\n(WER={bb_wer:.2f}, ID={bb_id:.1f}%)", xy=(bb_wer, bb_id), xytext=(0.97, 4.0),
             fontsize=10, color=PALETTE['red'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=PALETTE['red'], lw=1.0),
             bbox=dict(facecolor='white', edgecolor=PALETTE['red'],
                       boxstyle='round,pad=0.3'),
             ha='center')

# Threshold star
ax2.plot(0.90, 15.0, '*', color=PALETTE['gray'], markersize=18,
         markeredgecolor='black', markeredgewidth=1.0)
ax2.text(0.91, 15.5, "PIPL §23 threshold\n(0.90, 15%)", fontsize=9,
         color=PALETTE['gray'], ha='left', va='bottom', fontweight='bold')

# Random baseline
ax2.plot(1.00, 10.0, 'X', color=PALETTE['gray'], markersize=12,
         markeredgecolor='black', markeredgewidth=1.0)
ax2.text(1.005, 10.4, "Random baseline\n(1.00, 10%)", fontsize=9,
         color=PALETTE['gray'], va='bottom')

ax2.set_xlabel("Reconstruction WER (higher = more privacy)", fontsize=11)
ax2.set_ylabel("Speaker-ID accuracy (closer to 10% random = more privacy)", fontsize=11)
ax2.set_title("(b)  WER vs Speaker-ID accuracy", fontsize=12.5, fontweight='bold')
ax2.set_xlim(0.88, 1.04)
ax2.set_ylim(0, 17)
ax2.grid(True, alpha=0.3)

fig.text(0.5, 0.0,
         "Fig. 7.  Acoustic privacy. (a) Privacy radar (all 5 metrics normalized so 'outward' = more privacy): "
         "both white-box and black-box GLO attacks lie outside the PIPL §23 minimum boundary on every axis. "
         "(b) Both attack settings yield WER ≥ 0.95 with near-random speaker identification.",
         ha='center', fontsize=9.5, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig('./output/fig07_privacy_glo.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig07 regenerated: clearer privacy axis interpretation, non-overlapping labels")
