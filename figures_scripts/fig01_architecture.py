"""Figure 1: QAD-MultiGuard three-tier edge-cloud architecture."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import sci_style as sci

fig, ax = plt.subplots(figsize=(7.16, 4.0))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis("off")

# ---- Tier 1: On-device (left) ----
t1 = FancyBboxPatch((0.2, 0.4), 3.0, 5.2,
                    boxstyle="round,pad=0.06,rounding_size=0.15",
                    fc="#E8F1F8", ec="#1f77b4", lw=1.4)
ax.add_patch(t1)
ax.text(1.7, 5.30, "Tier 1: On-Device", fontsize=10, weight="bold",
        ha="center", color="#1f4060")
ax.text(1.7, 5.00, "Snapdragon 8 Gen 3 / Kirin 9000",
        fontsize=7.5, ha="center", color="#4c5b6b", style="italic")

# Sub-blocks for Tier 1
def small_block(x, y, w, h, text, color="#FFFFFF", ec="#1f77b4"):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=ec, lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.5)

small_block(0.45, 4.10, 2.50, 0.55, "SMS Feature Extractor (12-d)")
small_block(0.45, 3.40, 2.50, 0.55, "URL Feature Extractor (6-d)")
small_block(0.45, 2.55, 2.50, 0.70, r"Acoustic Embedder $F_v\in\mathbb{R}^{128}$" + "\n(non-invertible)")
small_block(0.45, 1.85, 2.50, 0.55, "Q4_K_M Student LLM (240 MB)")
small_block(0.45, 1.15, 2.50, 0.55, "Speculative Decoder ($\\gamma=5$)")
small_block(0.45, 0.55, 2.50, 0.50, "Local Risk Score $r_{\\rm local}$",
            color="#FFE9D6", ec="#ff7f0e")

ax.text(1.7, 0.15, "Raw PCM never leaves device  •  PIPL §23",
        ha="center", fontsize=6.5, style="italic", color="#a04000")

# ---- Tier 2: Cloud (middle) ----
t2 = FancyBboxPatch((3.7, 0.4), 2.8, 5.2,
                    boxstyle="round,pad=0.06,rounding_size=0.15",
                    fc="#FFF4E6", ec="#ff7f0e", lw=1.4)
ax.add_patch(t2)
ax.text(5.1, 5.30, "Tier 2: Cloud", fontsize=10, weight="bold",
        ha="center", color="#8a4a00")
ax.text(5.1, 5.00, "Blackwell GPU + vLLM",
        fontsize=7.5, ha="center", color="#5c4030", style="italic")

small_block(3.90, 4.10, 2.4, 0.55, "NVFP4 Server LLM (248 MB)",
            color="#FFFFFF", ec="#ff7f0e")
small_block(3.90, 3.40, 2.4, 0.55, "BF16 Teacher (self-distillation)",
            color="#FFFFFF", ec="#ff7f0e")
small_block(3.90, 2.55, 2.4, 0.70, "Chain-of-Thought Inference\n(CoT, $\\alpha=0.86$)",
            color="#FFFFFF", ec="#ff7f0e")
small_block(3.90, 1.85, 2.4, 0.55, "Cross-Modal Verifier",
            color="#FFFFFF", ec="#ff7f0e")
small_block(3.90, 1.15, 2.4, 0.55, "LoRA-QAD Incremental Update",
            color="#FFFFFF", ec="#ff7f0e")
small_block(3.90, 0.55, 2.4, 0.50, "Cloud Risk Score $r_{\\rm cloud}$",
            color="#FFE9D6", ec="#ff7f0e")

# ---- Tier 3: Fusion (right) ----
t3 = FancyBboxPatch((7.0, 0.4), 2.8, 5.2,
                    boxstyle="round,pad=0.06,rounding_size=0.15",
                    fc="#E8F5E9", ec="#2ca02c", lw=1.4)
ax.add_patch(t3)
ax.text(8.4, 5.30, "Tier 3: Fusion", fontsize=10, weight="bold",
        ha="center", color="#1e5a23")
ax.text(8.4, 5.00, "On-Device, < 1 ms",
        fontsize=7.5, ha="center", color="#3a5240", style="italic")

# Inputs
small_block(7.2, 4.10, 2.4, 0.55, "$r_{\\rm text}$  +  $r_{\\rm audio}$",
            color="#FFFFFF", ec="#2ca02c")
small_block(7.2, 3.40, 2.4, 0.55, "$r_{\\rm url}$   +   $r_{\\rm meta}$",
            color="#FFFFFF", ec="#2ca02c")
small_block(7.2, 2.55, 2.4, 0.70,
            "L-BFGS Sigmoid Fusion\n$r=\\sigma(\\mathbf{w}^\\top\\mathbf{r}+b)$",
            color="#FFFFFF", ec="#2ca02c")
small_block(7.2, 1.85, 2.4, 0.55, "Risk Classifier (3 levels)",
            color="#FFFFFF", ec="#2ca02c")
small_block(7.2, 1.15, 2.4, 0.55, "User UI Renderer",
            color="#FFFFFF", ec="#2ca02c")
small_block(7.2, 0.55, 2.4, 0.50, "Final Decision",
            color="#FFE9D6", ec="#ff7f0e")

# ---- Arrows: T1 -> T2 and T1 -> T3, T2 -> T3 ----
ax.annotate("", xy=(3.65, 3.0), xytext=(2.99, 3.0),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#1f77b4"))
ax.text(3.32, 3.15, r"$F_v$, text, URL", fontsize=6.8, ha="center",
        color="#1f4060")
ax.text(3.32, 2.80, "(features only)", fontsize=6.0, ha="center",
        color="#1f4060", style="italic")

ax.annotate("", xy=(6.95, 3.0), xytext=(6.31, 3.0),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#ff7f0e"))
ax.text(6.63, 3.15, r"$r_{\rm cloud}$", fontsize=6.8, ha="center",
        color="#8a4a00")

# Bottom: latency annotation
ax.text(5.0, -0.05, "End-to-end P50 = 268 ms  (Snapdragon 8 Gen 3, $n=5\\!,\\!000$, 30-day deployment)",
        ha="center", fontsize=7.5, weight="bold", color="#222")

sci.save(fig, "fig01_architecture.png", w=7.16, h=4.0)
print("Saved fig01_architecture.png")
