"""
Figure 9: QAD-MultiGuard training pipeline & schedule.

Layout:
  (a) Swim-lane pipeline. Teacher (top) and Student (bottom) lanes; lane labels
      live OUTSIDE the lanes on the left margin so the lane interiors are clean.
      Reading left → right shows the 3 training stages:
        \textcircled{1} Forward pass   — each lane produces its logits (p_T / p_S)
        \textcircled{2} KL loss        — single block in the middle, fed by both lanes
        \textcircled{3} Backprop       — only Student lane is updated; OV-Freeze tagged
      No crossing arrows; the only arrows are: input fork → lanes, lanes → KL,
      KL → student-update.

  (b) Training schedule, stacked panels sharing the x-axis (step 0–2000):
        Top    : phase ribbon (3 colored segments)
        Middle : LR curve with peak + OVF-start markers, both labeled in
                 numerically empty regions
        Bottom : OV-Freeze on/off status bar
      All numerical labels read directly off the shared x-axis.

Data: safety_data (TOTAL_STEPS=2000, WARMUP_STEPS=100, OVF_RATIO=0.30).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import sci_style as sci

# ─── Training constants (mirror QADConfig) ────────────────────────────────────
TOTAL_STEPS  = 2000
WARMUP_STEPS = 100
OVF_RATIO    = 0.30
OVF_START    = int(TOTAL_STEPS * (1 - OVF_RATIO))     # = 1400
PEAK_LR      = 1.0                                    # display as ×10⁻⁵

# ─── Colors ───────────────────────────────────────────────────────────────────
C_TEACHER = "#1f77b4"
C_STUDENT = "#ff7f0e"
C_KL      = "#2ca02c"
C_OVF     = "#cc5500"
C_WARMUP  = "#9467bd"
C_BG_T    = "#EAF2FA"
C_BG_S    = "#FFF1E0"

# ─── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.8, 4.6))
gs_outer = fig.add_gridspec(
    1, 2, width_ratios=[1.05, 1.10], wspace=0.18,
    left=0.06, right=0.97, top=0.93, bottom=0.10,
)

# ════════════════════════════════════════════════════════════════════════════
# Panel (a): swim-lane pipeline
# ════════════════════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs_outer[0])
ax_a.set_xlim(0, 11.5)
ax_a.set_ylim(0, 10)
ax_a.axis("off")
ax_a.set_title("(a) Pipeline:  forward $\\rightarrow$ loss $\\rightarrow$ backprop",
               weight="bold", fontsize=10, loc="left", x=0.0)

# ── Lane geometry ──
# Left margin (1.6 wide) hosts the lane labels as horizontal text;
# lane interiors then run from x=1.6 to x=11.4 cleanly.
LANE_X0, LANE_X1 = 1.65, 11.30
LANE_T_Y         = 6.0
LANE_S_Y         = 1.5
LANE_HEIGHT      = 2.6

# Lane backgrounds
ax_a.add_patch(Rectangle((LANE_X0, LANE_T_Y), LANE_X1 - LANE_X0, LANE_HEIGHT,
                          facecolor=C_BG_T, edgecolor=C_TEACHER, lw=1.2))
ax_a.add_patch(Rectangle((LANE_X0, LANE_S_Y), LANE_X1 - LANE_X0, LANE_HEIGHT,
                          facecolor=C_BG_S, edgecolor=C_STUDENT, lw=1.2))

# Lane labels — horizontal, placed in the LEFT MARGIN (x=0.1 to 1.55)
# Two lines per label, stacked, centered vertically with each lane.
ax_a.text(0.85, LANE_T_Y + LANE_HEIGHT / 2 + 0.30, "BF16",
          ha="center", va="center", fontsize=9.5, weight="bold",
          color=C_TEACHER)
ax_a.text(0.85, LANE_T_Y + LANE_HEIGHT / 2, "Teacher",
          ha="center", va="center", fontsize=9.5, weight="bold",
          color=C_TEACHER)
ax_a.text(0.85, LANE_T_Y + LANE_HEIGHT / 2 - 0.40, "(frozen)",
          ha="center", va="center", fontsize=7.5, style="italic",
          color=C_TEACHER)

ax_a.text(0.85, LANE_S_Y + LANE_HEIGHT / 2 + 0.30, "NVFP4",
          ha="center", va="center", fontsize=9.5, weight="bold",
          color=C_STUDENT)
ax_a.text(0.85, LANE_S_Y + LANE_HEIGHT / 2, "Student",
          ha="center", va="center", fontsize=9.5, weight="bold",
          color=C_STUDENT)
ax_a.text(0.85, LANE_S_Y + LANE_HEIGHT / 2 - 0.40, "(trainable)",
          ha="center", va="center", fontsize=7.5, style="italic",
          color=C_STUDENT)

# Stage boundaries — give stage 2 (KL loss) enough width for its label
STAGE_BOUNDS = [LANE_X0, 4.85, 7.95, LANE_X1]
STAGE_LABELS = ["1. Forward pass", "2. KL loss", "3. Backprop"]

# Stage dividers (dashed verticals inside lanes only)
for sx in STAGE_BOUNDS[1:-1]:
    ax_a.plot([sx, sx],
              [LANE_S_Y + 0.1, LANE_T_Y + LANE_HEIGHT - 0.1],
              color="#aaa", lw=0.6, ls=(0, (3, 3)), zorder=1)

# Stage labels ABOVE the teacher lane
STAGE_LBL_Y = LANE_T_Y + LANE_HEIGHT + 0.45
for sx0, sx1, lbl in zip(STAGE_BOUNDS[:-1], STAGE_BOUNDS[1:], STAGE_LABELS):
    ax_a.text((sx0 + sx1) / 2, STAGE_LBL_Y, lbl,
              ha="center", va="center", fontsize=8.3, weight="bold",
              color="#222")

# ── STAGE 1: Forward pass ─────────────────────────────────────────────────────
# Teacher content
T_X = 2.10
ax_a.add_patch(FancyBboxPatch((T_X, LANE_T_Y + 0.30), 2.40, 2.00,
                               boxstyle="round,pad=0.05,rounding_size=0.08",
                               fc="white", ec=C_TEACHER, lw=1.0))
ax_a.text(T_X + 1.20, LANE_T_Y + 1.85, "Qwen2.5-0.5B BF16",
          ha="center", va="center", fontsize=7.8, weight="bold",
          color=C_TEACHER)
ax_a.text(T_X + 1.20, LANE_T_Y + 1.40, "forward pass",
          ha="center", va="center", fontsize=7.3, color=C_TEACHER)
ax_a.plot([T_X + 0.20, T_X + 2.20], [LANE_T_Y + 1.10, LANE_T_Y + 1.10],
          color="#bbb", lw=0.5, ls="--")
ax_a.text(T_X + 1.20, LANE_T_Y + 0.70, r"output:  $p_T(y|x)$",
          ha="center", va="center", fontsize=8.5, weight="bold",
          color=C_TEACHER)

# Student content — content layout INVERTED so "output: p_S" is at TOP of box
# (since arrow exits upward toward the KL box that sits ABOVE the student lane)
S_X = T_X
ax_a.add_patch(FancyBboxPatch((S_X, LANE_S_Y + 0.30), 2.40, 2.00,
                               boxstyle="round,pad=0.05,rounding_size=0.08",
                               fc="white", ec=C_STUDENT, lw=1.0))
ax_a.text(S_X + 1.20, LANE_S_Y + 1.95, r"output:  $p_S(y|x)$",
          ha="center", va="center", fontsize=8.5, weight="bold",
          color=C_STUDENT)
ax_a.plot([S_X + 0.20, S_X + 2.20], [LANE_S_Y + 1.55, LANE_S_Y + 1.55],
          color="#bbb", lw=0.5, ls="--")
ax_a.text(S_X + 1.20, LANE_S_Y + 1.15, "Qwen2.5-0.5B NVFP4",
          ha="center", va="center", fontsize=7.8, weight="bold",
          color=C_STUDENT)
ax_a.text(S_X + 1.20, LANE_S_Y + 0.70, "(Q4_K_M variant)",
          ha="center", va="center", fontsize=7.3, color=C_STUDENT)

# Input badge — placed in the narrow gap between lane labels (x≈1.65-2.05)
# Actually with the new layout the lane label fills x=0.1-1.6 and lane starts at 1.65,
# so the input has to live ABOVE the gap. Move it to the strip between lanes (y≈4.9-5.3).
INPUT_X = 1.20
ax_a.add_patch(FancyBboxPatch((INPUT_X, 4.65), 0.80, 0.70,
                               boxstyle="round,pad=0.04,rounding_size=0.06",
                               fc="white", ec="#222", lw=0.9))
ax_a.text(INPUT_X + 0.40, 5.00, "Input\n$x$", ha="center", va="center",
          fontsize=7.5, linespacing=1.1)

# Fork from input into both lanes (curves up and down, never crossing)
ax_a.annotate("",
              xy=(T_X - 0.05, LANE_T_Y + 1.30),
              xytext=(INPUT_X + 0.80, 5.15),
              arrowprops=dict(arrowstyle="->", lw=1.0, color=C_TEACHER,
                              connectionstyle="arc3,rad=-0.25"))
ax_a.annotate("",
              xy=(S_X - 0.05, LANE_S_Y + 1.30),
              xytext=(INPUT_X + 0.80, 4.85),
              arrowprops=dict(arrowstyle="->", lw=1.0, color=C_STUDENT,
                              connectionstyle="arc3,rad=0.25"))

# ── STAGE 2: KL loss (between the lanes, centered horizontally in stage 2) ───
KL_CX = (STAGE_BOUNDS[1] + STAGE_BOUNDS[2]) / 2
KL_W  = 2.40
KL_H  = 1.30
KL_X  = KL_CX - KL_W / 2
KL_Y  = (LANE_T_Y + LANE_S_Y + LANE_HEIGHT) / 2 - KL_H / 2

ax_a.add_patch(FancyBboxPatch((KL_X, KL_Y), KL_W, KL_H,
                               boxstyle="round,pad=0.05,rounding_size=0.10",
                               fc="white", ec=C_KL, lw=1.5))
ax_a.text(KL_CX, KL_Y + KL_H - 0.32, r"$\mathcal{L}_{\rm QAD}$",
          ha="center", va="center", fontsize=12,
          color=C_KL, weight="bold")
ax_a.text(KL_CX, KL_Y + KL_H - 0.75,
          r"$=D_{\rm KL}\!\left(p_T\,\|\,p_S\right)$",
          ha="center", va="center", fontsize=8.5, color=C_KL)
ax_a.text(KL_CX, KL_Y + 0.20, "Pure KL  @ $T{=}1.0$",
          ha="center", va="center", fontsize=6.8, style="italic",
          color="#1e5a23")

# Arrows: teacher output (bottom of teacher box) → KL  (downward);
#         student output (top of student box) → KL  (upward)
TX_RIGHT = T_X + 2.40
SX_RIGHT = S_X + 2.40
ax_a.annotate("",
              xy=(KL_X + 0.5, KL_Y + KL_H),
              xytext=(TX_RIGHT, LANE_T_Y + 0.70),
              arrowprops=dict(arrowstyle="->", lw=1.1, color=C_TEACHER))
ax_a.annotate("",
              xy=(KL_X + 0.5, KL_Y),
              xytext=(SX_RIGHT, LANE_S_Y + 1.95),
              arrowprops=dict(arrowstyle="->", lw=1.1, color=C_STUDENT))

# ── STAGE 3: Backprop (only updates student) ─────────────────────────────────
# Teacher lane: explicit "no update" tag (helps reader understand asymmetry)
NU_X = STAGE_BOUNDS[2] + 0.20
NU_W = STAGE_BOUNDS[3] - NU_X - 0.20
ax_a.add_patch(FancyBboxPatch((NU_X, LANE_T_Y + 0.65), NU_W, 1.30,
                               boxstyle="round,pad=0.05,rounding_size=0.08",
                               fc="#F4F4F4", ec="#999", lw=0.8, ls=":"))
ax_a.text(NU_X + NU_W / 2, LANE_T_Y + 1.50, "no update",
          ha="center", va="center", fontsize=8.5,
          color="#666", weight="bold")
ax_a.text(NU_X + NU_W / 2, LANE_T_Y + 1.10, "(frozen weights)",
          ha="center", va="center", fontsize=7,
          color="#666", style="italic")

# Student lane: update equation + OV-Freeze sub-tag
UP_X = STAGE_BOUNDS[2] + 0.20
UP_W = STAGE_BOUNDS[3] - UP_X - 0.20
ax_a.add_patch(FancyBboxPatch((UP_X, LANE_S_Y + 0.30), UP_W, 2.00,
                               boxstyle="round,pad=0.05,rounding_size=0.08",
                               fc="white", ec=C_STUDENT, lw=1.0))
ax_a.text(UP_X + UP_W / 2, LANE_S_Y + 1.95,
          r"$\theta_S \leftarrow \theta_S - \eta\,\nabla\theta_S$",
          ha="center", va="center", fontsize=9, color=C_STUDENT)
ax_a.text(UP_X + UP_W / 2, LANE_S_Y + 1.55,
          "(all proj layers receive gradient)",
          ha="center", va="center", fontsize=7,
          color=C_STUDENT, style="italic")
# OV-Freeze regulator sub-tag (dashed orange) — placed at bottom of stage-3
ax_a.add_patch(FancyBboxPatch((UP_X + 0.10, LANE_S_Y + 0.45), UP_W - 0.20, 0.85,
                               boxstyle="round,pad=0.03,rounding_size=0.06",
                               fc="#FFE9D6", ec=C_OVF, lw=0.9, ls="--"))
ax_a.text(UP_X + UP_W / 2, LANE_S_Y + 1.05,
          "+ OV-Freeze  (last 30% steps)",
          ha="center", va="center", fontsize=7.3,
          color=C_OVF, weight="bold")
ax_a.text(UP_X + UP_W / 2, LANE_S_Y + 0.70,
          r"rescales $W_{q,k,v,o}$ to match BF16 variance",
          ha="center", va="center", fontsize=6.8, color=C_OVF)

# Gradient arrow: KL → student update block (single clean curve)
ax_a.annotate("",
              xy=(UP_X - 0.05, LANE_S_Y + LANE_HEIGHT - 0.45),
              xytext=(KL_X + KL_W, KL_Y + KL_H * 0.45),
              arrowprops=dict(arrowstyle="->", lw=1.4, color=C_OVF,
                              connectionstyle="arc3,rad=0.20"))
# Place ∇θ_S label ABOVE the curve mid-point, in clear space
ax_a.text(KL_X + KL_W + 0.55, KL_Y + KL_H * 0.85,
          r"$\nabla\theta_S$",
          fontsize=10.5, color=C_OVF, weight="bold", ha="center",
          bbox=dict(facecolor="white", edgecolor="none", pad=1))

# ════════════════════════════════════════════════════════════════════════════
# Panel (b): training schedule
# ════════════════════════════════════════════════════════════════════════════
gs_b = gs_outer[1].subgridspec(
    3, 1, height_ratios=[0.55, 2.6, 0.45], hspace=0.18,
)
ax_b_phase = fig.add_subplot(gs_b[0])
ax_b_lr    = fig.add_subplot(gs_b[1], sharex=ax_b_phase)
ax_b_ovf   = fig.add_subplot(gs_b[2], sharex=ax_b_phase)

for ax in (ax_b_phase, ax_b_ovf):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False)

ax_b_phase.set_title("(b) Training schedule  (2000 steps total)",
                     weight="bold", fontsize=10, loc="left", x=0.0)
ax_b_phase.set_xlim(0, TOTAL_STEPS)
ax_b_phase.set_ylim(-0.2, 1.05)
ax_b_phase.tick_params(labelbottom=False)

# ── Phase ribbon (TOP) ─────────────────────────────────────────────────────
phases = [
    (0,            WARMUP_STEPS,  "Warm-up",       C_WARMUP, "#F4ECFA"),
    (WARMUP_STEPS, OVF_START,     "Cosine decay",  C_TEACHER, "#E8F1F8"),
    (OVF_START,    TOTAL_STEPS,   "+ OV-Freeze",   C_OVF,    "#FFE9D6"),
]
for x0, x1, label, ec, fc in phases:
    ax_b_phase.add_patch(Rectangle((x0, 0.20), x1 - x0, 0.65,
                                    facecolor=fc, edgecolor=ec, linewidth=1.2))
    if (x1 - x0) / TOTAL_STEPS > 0.10:
        ax_b_phase.text((x0 + x1) / 2, 0.525, label,
                         ha="center", va="center", fontsize=8,
                         color=ec, weight="bold")
    else:
        # Warm-up is too narrow → label below ribbon with leader line
        ax_b_phase.annotate(
            label,
            xy=((x0 + x1) / 2, 0.20),
            xytext=((x0 + x1) / 2 + 70, -0.10),
            fontsize=7.5, color=ec, weight="bold",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.6, color=ec),
        )

# ── LR curve (MIDDLE) ──────────────────────────────────────────────────────
steps = np.arange(0, TOTAL_STEPS + 1)
lr = np.where(
    steps < WARMUP_STEPS,
    PEAK_LR * steps / WARMUP_STEPS,
    PEAK_LR * 0.5 * (1 + np.cos(np.pi * (steps - WARMUP_STEPS)
                                / (TOTAL_STEPS - WARMUP_STEPS))),
)

ax_b_lr.axvspan(0, WARMUP_STEPS, color="#F4ECFA", alpha=0.5, zorder=0)
ax_b_lr.axvspan(WARMUP_STEPS, OVF_START, color="#E8F1F8", alpha=0.35, zorder=0)
ax_b_lr.axvspan(OVF_START, TOTAL_STEPS, color="#FFE9D6", alpha=0.5, zorder=0)

ax_b_lr.plot(steps, lr, color=C_TEACHER, lw=2.0, zorder=3)
ax_b_lr.set_ylabel(r"LR  ($\times 10^{-5}$)", fontsize=9, color=C_TEACHER)
ax_b_lr.tick_params(axis="y", labelcolor=C_TEACHER)
ax_b_lr.tick_params(axis="x", labelbottom=False)
ax_b_lr.set_xlim(0, TOTAL_STEPS)
ax_b_lr.set_ylim(0, 1.35)
ax_b_lr.set_yticks([0.0, 0.25, 0.50, 0.75, 1.00])
ax_b_lr.grid(True, alpha=0.25)
ax_b_lr.spines["top"].set_visible(False)
ax_b_lr.spines["right"].set_visible(False)

# Peak marker + label (text in empty upper-right region of warm-up curve)
ax_b_lr.plot([WARMUP_STEPS], [PEAK_LR], "o", ms=7,
             markerfacecolor=C_TEACHER, markeredgecolor="white",
             lw=1.5, zorder=5)
ax_b_lr.annotate(
    "peak  $1{\\times}10^{-5}$\n@ step 100",
    xy=(WARMUP_STEPS, PEAK_LR),
    xytext=(WARMUP_STEPS + 260, PEAK_LR + 0.15),
    fontsize=7.3, color=C_TEACHER, ha="left", va="bottom",
    arrowprops=dict(arrowstyle="->", lw=0.7, color=C_TEACHER),
)

# OVF-start marker + label
lr_at_ovf = float(lr[OVF_START])
ax_b_lr.plot([OVF_START], [lr_at_ovf], "o", ms=7,
             markerfacecolor=C_OVF, markeredgecolor="white",
             lw=1.5, zorder=5)
ax_b_lr.annotate(
    f"$\\,{lr_at_ovf:.2f}{{\\times}}10^{{-5}}$\n@ OVF start (step 1400)",
    xy=(OVF_START, lr_at_ovf),
    xytext=(OVF_START - 220, lr_at_ovf + 0.42),
    fontsize=7.3, color=C_OVF, ha="center", va="bottom",
    arrowprops=dict(arrowstyle="->", lw=0.7, color=C_OVF),
)

# Token-budget annotation in bottom-right empty area
ax_b_lr.text(0.97, 0.06,
             "Total: 0.5 B tokens  ($\\approx$ 1.7% of SFT corpus)",
             transform=ax_b_lr.transAxes, ha="right", va="bottom",
             fontsize=7.3, color="#444", style="italic",
             bbox=dict(facecolor="white", edgecolor="none", pad=2))

# ── OV-Freeze status bar (BOTTOM) ──────────────────────────────────────────
ax_b_ovf.set_xlim(0, TOTAL_STEPS)
ax_b_ovf.set_ylim(0, 1)

ax_b_ovf.add_patch(Rectangle((0, 0.20), OVF_START, 0.60,
                              facecolor="#EEEEEE", edgecolor="#999", lw=0.7))
ax_b_ovf.text(OVF_START / 2, 0.50, "OV-Freeze OFF",
              ha="center", va="center", fontsize=7.3,
              color="#666", style="italic")

ax_b_ovf.add_patch(Rectangle((OVF_START, 0.20), TOTAL_STEPS - OVF_START, 0.60,
                              facecolor=C_OVF, edgecolor=C_OVF, lw=0.7, alpha=0.9))
ax_b_ovf.text((OVF_START + TOTAL_STEPS) / 2, 0.50, "ON  (last 30%)",
              ha="center", va="center", fontsize=7.3,
              color="white", weight="bold")

ax_b_ovf.text(-0.012, 0.50, "OVF",
              transform=ax_b_ovf.transAxes,
              ha="right", va="center", fontsize=8.5,
              color=C_OVF, weight="bold")

ax_b_ovf.set_xlabel("Training step", fontsize=9)
ax_b_ovf.tick_params(labelbottom=True, bottom=True)
ax_b_ovf.set_xticks([0, WARMUP_STEPS, 500, 1000, OVF_START, TOTAL_STEPS])

# ─── Save ──────────────────────────────────────────────────────────────────────
sci.save(fig, "fig09_qad_pipeline.png", w=8.8, h=4.6)
print("Saved fig09_qad_pipeline.png")
