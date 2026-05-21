"""
Fig 1: QAD-MultiGuard three-tier edge-cloud architecture
SCI journal quality — unified color families per tier, clear partitions, unambiguous arrows.
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

os.makedirs('./output', exist_ok=True)

# ── Unified SCI color palette ──────────────────────────────────────
C = {
    # Tier 1 — On-Device (Blue family)
    't1_bg':     '#E8F0FE',
    't1_edge':   '#1A73E8',
    't1_fill':   '#D2E3FC',
    't1_text':   '#174EA6',
    't1_accent': '#1967D2',
    # Tier 2 — Cloud (Orange/Amber family)
    't2_bg':     '#FEF7E0',
    't2_edge':   '#E37400',
    't2_fill':   '#FCE8C8',
    't2_text':   '#B45F06',
    't2_accent': '#E37400',
    # Tier 3 — Fusion (Green family)
    't3_bg':     '#E6F4EA',
    't3_edge':   '#1E8E3E',
    't3_fill':   '#CEEAD6',
    't3_text':   '#137333',
    't3_accent': '#188038',
    # Neutral
    'navy':      '#1F3864',
    'gray':      '#595959',
    'lgray':     '#BFBFBF',
    'white':     '#FFFFFF',
    # Accent
    'red':       '#C00000',
    'lred':      '#FCE8E8',
    'alert':     '#D93025',
}

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.unicode_minus': False,
})

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

# Helper: draw a tier swimlane with left accent bar + vertical label
def draw_tier(ax, x, y, w, h, label, sublabel, accent_color, bg_color, edge_color):
    """Draw a tier background with left accent bar."""
    # Main background
    bg = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.04,rounding_size=0.06",
                         linewidth=1.5, edgecolor=edge_color, facecolor=bg_color, zorder=1)
    ax.add_patch(bg)
    # Left accent bar
    bar = FancyBboxPatch((x + 0.08, y + 0.2), 0.12, h - 0.4,
                          boxstyle="round,pad=0.02,rounding_size=0.03",
                          linewidth=0, facecolor=accent_color, alpha=0.7, zorder=2)
    ax.add_patch(bar)
    # Tier label (vertical, left of accent bar)
    ax.text(x + 0.14, y + h / 2, label,
            fontsize=10.5, fontweight='bold', color=accent_color,
            ha='center', va='center', rotation=90, zorder=3)
    # Sublabel below tier title
    ax.text(x + 1.0, y + h - 0.22, sublabel,
            fontsize=8, color=C['gray'], style='italic', zorder=3)


def draw_module(ax, x, y, w, h, title, subtitle, fill_color, edge_color, text_color):
    """Draw a rounded module box with title + subtitle."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.03,rounding_size=0.04",
                          linewidth=1.2, edgecolor=edge_color, facecolor=fill_color, zorder=4)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.18, title,
            ha='center', va='top', fontsize=9.5, fontweight='bold',
            color=text_color, zorder=5)
    ax.text(x + w/2, y + 0.18, subtitle,
            ha='center', va='center', fontsize=7.8,
            color=C['gray'], style='italic', zorder=5)


def draw_arrow(ax, x1, y1, x2, y2, color, lw=1.8, style='-', rad=0.0):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color,
                               connectionstyle=f"arc3,rad={rad}",
                               linestyle=style))


# ═══════════════════════════════════════════════════════════════════
# PRIVACY BANNER (top)
# ═══════════════════════════════════════════════════════════════════
banner_y0, banner_h = 9.05, 0.55
banner = FancyBboxPatch((0.3, banner_y0), 15.4, banner_h,
                         boxstyle="round,pad=0.04,rounding_size=0.06",
                         linewidth=1.8, edgecolor=C['red'], facecolor=C['lred'], zorder=6)
ax.add_patch(banner)
ax.text(8.0, banner_y0 + banner_h/2,
        "Privacy Constraint (PIPL §23): raw audio NEVER leaves device — only 128-d non-invertible $\\mathbf{F}_v$",
        ha='center', va='center', fontsize=10.5, fontweight='bold',
        color=C['red'], style='italic', zorder=7)

# ═══════════════════════════════════════════════════════════════════
# INPUT MODALITIES
# ═══════════════════════════════════════════════════════════════════
input_data = [
    (2.5, 8.55, "[ SMS ]"),
    (5.5, 8.55, "[ Voice Call ]"),
    (8.5, 8.55, "[ URL Links ]"),
    (11.5, 8.55, "[ Call Metadata ]"),
]
for x, y, t in input_data:
    ax.text(x, y, t, fontsize=9.5, ha='center', va='center',
            family='monospace', color=C['navy'], fontweight='bold')
# Input arrows down to Tier 1
for x, _, _ in input_data:
    draw_arrow(ax, x, 8.30, x, 7.65, C['navy'], lw=1.4)

# ═══════════════════════════════════════════════════════════════════
# TIER 1 — On-Device Detection
# ═══════════════════════════════════════════════════════════════════
t1_x, t1_y, t1_w, t1_h = 0.4, 5.55, 15.2, 2.1
draw_tier(ax, t1_x, t1_y, t1_w, t1_h,
          "Tier 1\nOn-Device\nDetection",
          "P50: 280 ms · P99: 360 ms · ARM/Snapdragon 8 Gen 3 · 240 MB",
          C['t1_accent'], C['t1_bg'], C['t1_edge'])

# Tier 1 modules — all in unified blue color family
t1_mods = [
    (0.8, 5.8,  2.6, 1.2,  "SMS Module",     "12-d feature extraction"),
    (3.7, 5.8,  2.4, 1.2,  "URL Module",      "6-d structural features"),
    (6.4, 5.8,  2.6, 1.2,  "Acoustic Module", "$F_v = [f_{mfcc}; W\\bar{h}_w]$\n128-d non-invertible"),
    (9.3, 5.8,  2.8, 1.2,  "Student LLM",     "Qwen2.5-0.5B (Q4_K_M)\nspec-decode α=0.86 · 3.49×"),
    (12.4, 5.8, 2.8, 1.2,  "Local Risk Score","on-device fast path\n(high-confidence bypass)"),
]
for x, y, w, h, title, sub in t1_mods:
    draw_module(ax, x, y, w, h, title, sub, C['t1_fill'], C['t1_edge'], C['t1_text'])

# ── F_v DP upload arrow (from Acoustic Module down to Tier 2) ────
draw_arrow(ax, 7.7, 5.75, 7.7, 4.45, C['t1_accent'], lw=2.0)
ax.text(7.9, 5.05, "$F_v$ upload (ε=1.5 DP)",
        fontsize=8.5, fontweight='bold', color=C['t1_accent'], zorder=8)

# ── Fast-path arrow (from Local Risk Score to Tier 3, skipping Tier 2) ────
draw_arrow(ax, 13.8, 5.75, 13.8, 2.15, C['t1_edge'], lw=1.4, style='dashed', rad=0.25)
ax.text(14.2, 3.9, "fast path\n(high conf.)",
        fontsize=8, color=C['t1_edge'], style='italic', zorder=8)

# ═══════════════════════════════════════════════════════════════════
# TIER 2 — Cloud Inference
# ═══════════════════════════════════════════════════════════════════
t2_x, t2_y, t2_w, t2_h = 0.4, 3.25, 15.2, 1.7
draw_tier(ax, t2_x, t2_y, t2_w, t2_h,
          "Tier 2\nCloud\nInference",
          "Async · conditional · NVFP4 on Blackwell GPU · vLLM · 248 MB",
          C['t2_accent'], C['t2_bg'], C['t2_edge'])

# Tier 2 modules — unified orange family
t2_mods = [
    (0.7, 3.45,  3.4, 1.0,  "CoT Reasoning",       "step-by-step reasoning\nchain-of-thought"),
    (4.3, 3.45,  3.6, 1.0,  "NVFP4 Server LLM",     "Qwen2.5-0.5B (248 MB)\n99.1% recovery · 4× compression"),
    (8.1, 3.45,  3.2, 1.0,  "OV-Freeze Regularizer", "q/k/v/o-proj variance\ncorrection · 30% step ratio"),
    (11.5, 3.45, 3.8, 1.0,  "Cross-Modal Verifier",  "consistency check across\nall four input modalities"),
]
for x, y, w, h, title, sub in t2_mods:
    draw_module(ax, x, y, w, h, title, sub, C['t2_fill'], C['t2_edge'], C['t2_text'])

# Arrows: Tier 1 bottom → Tier 2 top (only for modules that feed cloud)
# Acoustic, Student LLM, and Local Risk Score all feed into Tier 2
for x_src in [2.1, 4.9, 7.7, 10.7, 13.8]:
    draw_arrow(ax, x_src, 5.5, x_src, 5.0, C['t2_accent'], lw=1.4)

# ═══════════════════════════════════════════════════════════════════
# TIER 3 — Multimodal Risk Fusion
# ═══════════════════════════════════════════════════════════════════
t3_x, t3_y, t3_w, t3_h = 0.4, 0.75, 12.0, 2.15
draw_tier(ax, t3_x, t3_y, t3_w, t3_h,
          "Tier 3\nMultimodal\nFusion",
          "<1 ms · L-BFGS optimized · 5-fold CV verified (σ < 0.018)",
          C['t3_accent'], C['t3_bg'], C['t3_edge'])

# Tier 3: four risk scores
t3_scores = [
    (0.7, 0.95,  2.0, 1.2,  "$r_{\\mathrm{text}}$",   "$w$ = 0.40"),
    (2.9, 0.95,  2.0, 1.2,  "$r_{\\mathrm{audio}}$",  "$w$ = 0.30"),
    (5.1, 0.95,  2.0, 1.2,  "$r_{\\mathrm{url}}$",    "$w$ = 0.20"),
    (7.3, 0.95,  2.0, 1.2,  "$r_{\\mathrm{meta}}$",   "$w$ = 0.10"),
]
for x, y, w, h, title, sub in t3_scores:
    draw_module(ax, x, y, w, h, title, sub, C['t3_fill'], C['t3_edge'], C['t3_text'])

# ── Arrows: Tier 2 → Tier 3 ────────────────────────────────────
for x_src in [2.4, 4.9, 7.5]:
    draw_arrow(ax, x_src, 3.2, x_src, 2.95, C['t3_accent'], lw=1.4)

# ── σ-fusion block ─────────────────────────────────────────────
sigma_x, sigma_y, sigma_w, sigma_h = 9.6, 0.95, 4.0, 1.2
sigma = FancyBboxPatch((sigma_x, sigma_y), sigma_w, sigma_h,
                        boxstyle="round,pad=0.04,rounding_size=0.05",
                        linewidth=1.8, edgecolor=C['t3_edge'],
                        facecolor=C['t3_edge'], alpha=0.88, zorder=5)
ax.add_patch(sigma)
ax.text(sigma_x + sigma_w/2, sigma_y + sigma_h * 0.65,
        r"$r = \sigma\left( \sum_m w_m \cdot r_m + b \right)$",
        ha='center', va='center', fontsize=11, fontweight='bold',
        color=C['white'], zorder=6)
ax.text(sigma_x + sigma_w/2, sigma_y + sigma_h * 0.30,
        "Decision: safe / medium / high",
        ha='center', va='center', fontsize=8.5,
        color=C['white'], style='italic', zorder=6)

# Arrows from four risk scores into sigma fusion
for i, (x, y, w, h, _, _) in enumerate(t3_scores):
    draw_arrow(ax, x + w, y + h/2, sigma_x, y + h/2,
               C['t3_accent'], lw=1.4)

# ═══════════════════════════════════════════════════════════════════
# ALERT OUTPUT
# ═══════════════════════════════════════════════════════════════════
draw_arrow(ax, 13.6, 1.55, 14.5, 1.55, C['alert'], lw=2.5)
alert = FancyBboxPatch((14.5, 1.15), 0.9, 0.8,
                        boxstyle="round,pad=0.04,rounding_size=0.05",
                        linewidth=2.0, edgecolor=C['alert'],
                        facecolor=C['alert'], zorder=6)
ax.add_patch(alert)
ax.text(14.95, 1.55, "ALERT", ha='center', va='center',
        fontsize=11, fontweight='bold', color=C['white'], zorder=7)

# ── Feedback arrow (from ALERT back to Tier 1) ─────────────────
draw_arrow(ax, 14.95, 1.0, 14.95, 5.5,
           C['gray'], lw=1.0, style='dashed', rad=0.4)
ax.text(15.4, 3.2, "online\nfeedback",
        fontsize=7.5, color=C['gray'], style='italic', rotation=90, zorder=8)

# ═══════════════════════════════════════════════════════════════════
# CAPTION
# ═══════════════════════════════════════════════════════════════════
fig.text(0.5, 0.01,
         "Fig. 1.  QAD-MultiGuard three-tier edge-cloud collaborative architecture for telecom fraud detection. "
         "Tier 1 performs on-device detection with speculative decoding (α=0.86, γ=5, 3.49× measured on Snapdragon 8 Gen 3). "
         "Tier 2 provides conditional cloud inference with NVFP4 QAD + OV-Freeze (99.1% F1 recovery). "
         "Tier 3 fuses all modalities via L-BFGS optimized σ-fusion (σ < 0.018 across 5-fold CV). "
         "Federated learning is future work (§5.2) and not part of the current Tier 2.",
         ha='center', fontsize=9.5, style='italic')

plt.tight_layout(rect=[0, 0.025, 1, 0.99])
fig.savefig('./output/fig01_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('./output/fig01_architecture.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print("[OK] fig01 regenerated: tiered architecture with unified color families")
