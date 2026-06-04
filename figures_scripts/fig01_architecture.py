"""Figure 1: QAD-MultiGuard three-tier edge-cloud architecture (CN labels).

Comprehensive system diagram showing:
  - PIPL §23 privacy constraint banner (top)
  - Four input modalities (SMS / Voice Call / URL Links / Call Metadata)
  - Tier 1 (Edge): 5 on-device modules + Fast Path
  - Tier 2 (Cloud): CoT inference + NVFP4 model + OV-Freeze + Federated aggregator
  - Tier 3 (Fusion): 4 per-modality risk scores + L-BFGS linear fusion + Alert
  - Cross-tier data flows: F_v upload (DP), Fast Path, Federated aggregation
  - Legend + abbreviation glossary (bottom)
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D

import sci_style as sci  # noqa: E402

# CJK font setup — MUST come AFTER sci_style import to override its DejaVu setting
mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                    "WenQuanYi Zen Hei", "DejaVu Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["mathtext.fontset"] = "stix"

# Color palette — match reference image
C_EDGE_FC   = "#FFFFFF"
C_EDGE_EC   = "#2ca02c"
C_EDGE_BG   = "#F0F9F4"
C_CLOUD_FC  = "#FFFFFF"
C_CLOUD_EC  = "#1f77b4"
C_CLOUD_BG  = "#F0F6FC"
C_FUSION_FC = "#FFFFFF"
C_FUSION_EC = "#ff7f0e"
C_FUSION_BG = "#FFF8F0"
C_ALERT     = "#d62728"
C_PRIVACY   = "#1f4060"
C_INPUT     = "#444"

fig, ax = plt.subplots(figsize=(11.0, 8.0))
ax.set_xlim(0, 22)
ax.set_ylim(0, 16)
ax.axis("off")

# ═══════════════════════════════════════════════════════════════════════
# Privacy banner (top)
# ═══════════════════════════════════════════════════════════════════════
banner = FancyBboxPatch((0.4, 15.1), 21.2, 0.7,
                        boxstyle="round,pad=0.04,rounding_size=0.10",
                        fc="#EAF2FB", ec=C_PRIVACY, lw=1.0)
ax.add_patch(banner)
ax.text(11.0, 15.45,
        "[隐私] 隐私约束(PIPL §23):原始语音数据不离开设备,仅上传 128 维不可逆声学特征向量 $F_v$(DP 脱敏)",
        ha="center", va="center", fontsize=10, color=C_PRIVACY, weight="bold")

# ═══════════════════════════════════════════════════════════════════════
# Input modalities row
# ═══════════════════════════════════════════════════════════════════════
input_box = FancyBboxPatch((0.4, 13.9), 21.2, 1.0,
                            boxstyle="round,pad=0.04,rounding_size=0.10",
                            fc="#FFFFFF", ec=C_PRIVACY, lw=1.0)
ax.add_patch(input_box)
ax.text(1.4, 14.4, "输入模态", fontsize=10, weight="bold", ha="center",
        va="center", color=C_PRIVACY)

# Four input icons + labels (use colored markers + text labels)
# Positions aligned so arrows to Tier-1 modules go straight down
inputs = [
    (4.85, "■", "短信(SMS)",              "#2ca02c"),  # → SMS module at x=3.10+W/2=4.85
    (8.65, "◆", "URL 链接(URL Links)",    "#9467bd"),  # → URL module at x=6.90+W/2=8.65
    (12.45,"▶", "语音通话(Voice Call)",    "#1f77b4"),  # → Acoustic module at x=10.70+W/2=12.45
    (16.25,"●", "通话元数据(Call Metadata)","#ff7f0e"), # → 端侧模型 at x=14.50+W/2=16.25
]
for x, icon, label, color in inputs:
    ax.text(x, 14.4, icon, fontsize=14, va="center", ha="center", color=color)
    ax.text(x + 0.30, 14.4, label, fontsize=9, va="center", ha="left",
            color=C_INPUT)

# ═══════════════════════════════════════════════════════════════════════
# TIER 1 — Edge (top tier)
# ═══════════════════════════════════════════════════════════════════════
t1 = FancyBboxPatch((0.4, 10.0), 21.2, 3.7,
                    boxstyle="round,pad=0.06,rounding_size=0.12",
                    fc=C_EDGE_BG, ec=C_EDGE_EC, lw=1.4)
ax.add_patch(t1)
ax.text(1.4, 13.2, "第一层:端侧检测", fontsize=10.5, weight="bold",
        ha="center", va="center", color=C_EDGE_EC)
ax.text(1.4, 12.85, "(EDGE)", fontsize=8.5, ha="center", va="center",
        color=C_EDGE_EC, style="italic")

# Tier-1 sidebar specs (left)
ax.text(0.55, 12.30, "• 延迟:~80–120 ms", fontsize=8, ha="left", va="center", color="#333")
ax.text(0.55, 11.95, "• 内存:< 240 MB RAM", fontsize=8, ha="left", va="center", color="#333")
ax.text(0.55, 11.60, "• 平台:Snapdragon 8 Gen 3", fontsize=8, ha="left", va="center", color="#333")

T1_Y = 10.30
T1_H = 2.50

# 5 sub-blocks for Tier 1
def tier_block(x, y, w, h, title_icon, title, subtitle, body_lines,
                ec=C_EDGE_EC, fc=C_EDGE_FC, icon_color="#222"):
    """Draw a tier-1/2/3 sub-block with icon, title, divider, body lines."""
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.0))
    # Title row
    ax.text(x + 0.3, y + h - 0.30, title_icon, fontsize=11, va="center", ha="center",
            color=icon_color)
    ax.text(x + w/2 + 0.15, y + h - 0.28, title, fontsize=9,
            weight="bold", ha="center", va="center", color="#222")
    if subtitle:
        ax.text(x + w/2 + 0.15, y + h - 0.58, subtitle, fontsize=7.5,
                ha="center", va="center", color="#666")
    # Divider line
    ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.78, y + h - 0.78],
            color="#bbb", lw=0.5, ls="--")
    # Body
    for i, line in enumerate(body_lines):
        ax.text(x + w/2, y + h - 1.08 - i * 0.30, line, fontsize=7.8,
                ha="center", va="center", color="#222")

# Tier-1 modules — use safe geometric markers (■ ◆ ● ▲) colored per module
mods_t1 = [
    (3.10, "■",  "#2ca02c", "SMS 模块",   "文本分析",      ["12 维文本", "特征提取"]),
    (6.90, "◆",  "#9467bd", "URL 模块",   "结构分析",     ["6 维结构", "特征提取"]),
    (10.70,"〰",  "#1f77b4", "声学模块",   "语音编码",      ["128 维不可逆声学向量 $F_v$", "(DP 脱敏扰动)"]),
    (14.50,"★",  "#cc5500", "端侧模型",   "(Q4_K_M Student)", ["Qwen2.5-0.5B",
                                                              "Spec-decode $\\alpha$=0.86",
                                                              "推理速度提升 ~ 3.5 ×"]),
    (18.30,"✓",  "#2ca02c", "本地风险评分","(Fast Path)",  ["高置信度情况下", "直接低延迟决策"]),
]
W_BLOCK = 3.50
for x, icon, ic_color, title, sub, body in mods_t1:
    tier_block(x, T1_Y, W_BLOCK, T1_H, icon, title, sub, body, icon_color=ic_color)

# Arrows from inputs to Tier-1 modules (straight downward, aligned)
input_xs    = [4.85, 8.65, 12.45, 16.25]      # input icon x positions
target_xs   = [4.85,                  # SMS  → SMS module
               8.65,                  # URL  → URL module
               12.45,                 # Voice → 声学模块 (acoustic)
               16.25]                 # Metadata → 端侧模型
for in_x, mod_x in zip(input_xs, target_xs):
    ax.annotate("", xy=(mod_x, T1_Y + T1_H), xytext=(in_x, 13.85),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#888"))

# ═══════════════════════════════════════════════════════════════════════
# Inter-tier flow labels (between T1 and T2)
# ═══════════════════════════════════════════════════════════════════════
# F_v upload arrow (DP)
ax.annotate("", xy=(10.70 + W_BLOCK/2, 9.10), xytext=(10.70 + W_BLOCK/2, 10.30),
            arrowprops=dict(arrowstyle="->", lw=1.2, color=C_CLOUD_EC))
ax.text(10.70 + W_BLOCK/2 - 0.05, 9.75, "$F_v$ 上传($\\epsilon=1.5$ DP)",
        fontsize=8, ha="center", va="center", color=C_CLOUD_EC, weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

# Async trigger from Tier-1 model → Tier-2
ax.annotate("", xy=(14.50 + W_BLOCK/2, 9.10), xytext=(14.50 + W_BLOCK/2, 10.30),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#888"))
ax.text(14.50 + W_BLOCK/2, 9.70, "异步条件触发",
        fontsize=8, ha="center", va="center", color="#333",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

# Fast path (Tier-1 → Tier-3 directly, bypasses Tier-2)
# Draw a long dashed green arrow from Tier-1 Fast-Path block down to Alert
ax.annotate("", xy=(20.6, 1.5), xytext=(18.30 + W_BLOCK/2, 10.30),
            arrowprops=dict(arrowstyle="->", lw=1.2, color=C_EDGE_EC,
                            linestyle=(0, (5, 3))))
ax.text(20.40, 9.10, "$r_{\\rm local}$", fontsize=8.5, ha="center",
        va="center", color=C_EDGE_EC, weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

# ═══════════════════════════════════════════════════════════════════════
# TIER 2 — Cloud (middle tier)
# ═══════════════════════════════════════════════════════════════════════
t2 = FancyBboxPatch((0.4, 5.30), 21.2, 3.80,
                    boxstyle="round,pad=0.06,rounding_size=0.12",
                    fc=C_CLOUD_BG, ec=C_CLOUD_EC, lw=1.4)
ax.add_patch(t2)
ax.text(1.4, 8.60, "第二层:云端推理", fontsize=10.5, weight="bold",
        ha="center", va="center", color=C_CLOUD_EC)
ax.text(1.4, 8.25, "(异步条件触发)", fontsize=8.5, ha="center", va="center",
        color=C_CLOUD_EC, style="italic")

# Tier-2 sidebar specs
ax.text(0.55, 7.65, "• 延迟:~200–600 ms (P95)", fontsize=8, ha="left", va="center", color="#333")
ax.text(0.55, 7.30, "• GPU:A100 80GB", fontsize=8, ha="left", va="center", color="#333")
ax.text(0.55, 6.95, "• 量化格式:NVFP4", fontsize=8, ha="left", va="center", color="#333")

# 4 sub-blocks for Tier 2
T2_Y = 5.60
T2_H = 2.50
mods_t2 = [
    (3.10, "▷",  "#1f77b4", "CoT 推理模块",   "链式推理",    ["分步骤风险分析"]),
    (7.90, "■",  "#1f77b4", "云端模型",       "(NVFP4 量化)", ["Qwen2.5-0.5B", "精度恢复率 99.1%"]),
    (12.70,"※",  "#1f77b4", "OV-Freeze 层",   "输出方差冻结",["修正 $q, k, v, o$-proj", "方差漂移"]),
    (17.50,"●",  "#1f77b4", "联邦聚合器",     "隐私保护聚合",["跨用户知识聚合", "不上传原始数据"]),
]
W_T2 = 4.40
for x, icon, ic_color, title, sub, body in mods_t2:
    tier_block(x, T2_Y, W_T2, T2_H, icon, title, sub, body, ec=C_CLOUD_EC,
               icon_color=ic_color)

# Arrows within Tier-2 (left to right chain): CoT → 云端模型 → OV-Freeze → 联邦
for x_from, x_to in [(3.10 + W_T2, 7.90), (7.90 + W_T2, 12.70),
                     (12.70 + W_T2, 17.50)]:
    ax.annotate("", xy=(x_to, T2_Y + T2_H/2), xytext=(x_from, T2_Y + T2_H/2),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#666"))

# ═══════════════════════════════════════════════════════════════════════
# Inter-tier flow T2 → T3
# ═══════════════════════════════════════════════════════════════════════
# r_cloud arrow (from cloud model)
ax.annotate("", xy=(7.90 + W_T2/2, 4.40), xytext=(7.90 + W_T2/2, 5.60),
            arrowprops=dict(arrowstyle="->", lw=1.2, color=C_CLOUD_EC))
ax.text(7.90 + W_T2/2 + 0.20, 5.00, "$r_{\\rm cloud}$",
        fontsize=9, ha="center", va="center", color=C_CLOUD_EC, weight="bold",
        bbox=dict(facecolor="white", edgecolor="none", pad=1))

# Federated knowledge feedback (dashed blue, from federated → fusion)
ax.annotate("", xy=(17.50 + W_T2/2, 4.40), xytext=(17.50 + W_T2/2, 5.60),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=C_CLOUD_EC,
                            linestyle=(0, (4, 2))))

# ═══════════════════════════════════════════════════════════════════════
# TIER 3 — Fusion (bottom tier)
# ═══════════════════════════════════════════════════════════════════════
t3 = FancyBboxPatch((0.4, 0.6), 19.4, 3.80,
                    boxstyle="round,pad=0.06,rounding_size=0.12",
                    fc=C_FUSION_BG, ec=C_FUSION_EC, lw=1.4)
ax.add_patch(t3)
ax.text(1.4, 3.90, "第三层:多模态风险融合", fontsize=10.5, weight="bold",
        ha="center", va="center", color=C_FUSION_EC)
ax.text(1.4, 3.55, "(EDGE)", fontsize=8.5, ha="center", va="center",
        color=C_FUSION_EC, style="italic")

# Tier-3 sidebar specs
ax.text(0.55, 2.90, "• 耗时:~1 ms", fontsize=8, ha="left", va="center", color="#333")
ax.text(0.55, 2.45, "• 算法:LBFGS 优化", fontsize=8, ha="left", va="center", color="#333")
ax.text(0.55, 2.10, "• 线性融合", fontsize=8, ha="left", va="center", color="#333")

# 4 risk-score input blocks
T3_Y_TOP = 2.30
T3_H_TOP = 1.80
risk_blocks = [
    (3.10,  "■", "#2ca02c", r"文本风险 $r_{\rm text}$",  "(来自第一层/第二层)", "权重 $w = 0.40$"),
    (7.00,  "〰", "#1f77b4", r"音频风险 $r_{\rm audio}$","(来自第一层/第二层)", "权重 $w = 0.30$"),
    (10.90, "◆", "#9467bd", r"URL 风险 $r_{\rm url}$",   "(来自第一层/第二层)", "权重 $w = 0.20$"),
    (14.80, "●", "#ff7f0e", r"元数据风险 $r_{\rm meta}$","(来自第一层)",       "权重 $w = 0.10$"),
]
W_T3 = 3.70
for x, icon, ic_color, title, sub, wt in risk_blocks:
    ax.add_patch(FancyBboxPatch((x, T3_Y_TOP), W_T3, T3_H_TOP,
                                boxstyle="round,pad=0.04,rounding_size=0.08",
                                fc=C_FUSION_FC, ec=C_FUSION_EC, lw=0.9))
    ax.text(x + 0.30, T3_Y_TOP + T3_H_TOP - 0.30, icon,
            fontsize=11, va="center", ha="center", color=ic_color)
    ax.text(x + W_T3/2 + 0.15, T3_Y_TOP + T3_H_TOP - 0.30, title,
            fontsize=9, weight="bold", ha="center", va="center", color="#222")
    ax.text(x + W_T3/2, T3_Y_TOP + T3_H_TOP - 0.65, sub,
            fontsize=7, ha="center", va="center", color="#666")
    ax.plot([x + 0.15, x + W_T3 - 0.15],
            [T3_Y_TOP + T3_H_TOP - 0.95, T3_Y_TOP + T3_H_TOP - 0.95],
            color="#bbb", lw=0.5, ls="--")
    ax.text(x + W_T3/2, T3_Y_TOP + T3_H_TOP - 1.40, wt,
            fontsize=8.5, ha="center", va="center", color="#cc5500",
            weight="bold")

# Fusion equation bar (bottom-center)
fb = FancyBboxPatch((3.10, 0.70), 15.40, 1.40,
                    boxstyle="round,pad=0.04,rounding_size=0.08",
                    fc="#FFF4E6", ec=C_FUSION_EC, lw=1.0)
ax.add_patch(fb)
ax.text(10.80, 1.70, "风险融合(线性加权)",
        fontsize=9.5, weight="bold", ha="center", va="center", color="#8a4a00")
ax.text(10.80, 1.30,
        r"$r = \sum_{m}(w_m \cdot r_m) + b$",
        fontsize=11, ha="center", va="center", color="#222")
ax.text(10.80, 0.92,
        "决策结果:Safe(安全)  /  Medium(中风险)  /  High(高风险)",
        fontsize=8, ha="center", va="center", color="#333")

# Arrows from each risk block down to fusion bar
for x, _, _, _, _, _ in risk_blocks:
    ax.annotate("", xy=(x + W_T3/2, 2.10), xytext=(x + W_T3/2, T3_Y_TOP),
                arrowprops=dict(arrowstyle="->", lw=0.9, color="#888"))

# ═══════════════════════════════════════════════════════════════════════
# Alert output (bottom-right)
# ═══════════════════════════════════════════════════════════════════════
alert = FancyBboxPatch((20.0, 0.70), 1.55, 1.55,
                       boxstyle="round,pad=0.04,rounding_size=0.10",
                       fc="#FFFFFF", ec=C_ALERT, lw=1.5)
ax.add_patch(alert)
ax.text(20.78, 1.85, "!", fontsize=22, ha="center", va="center",
        color=C_ALERT, weight="bold")
ax.text(20.78, 1.30, "风险告警", fontsize=9, weight="bold",
        ha="center", va="center", color=C_ALERT)
ax.text(20.78, 0.95, "(ALERT)", fontsize=7.5, ha="center", va="center",
        color=C_ALERT, style="italic")

# Arrow fusion → alert (red, bold)
ax.annotate("", xy=(20.0, 1.40), xytext=(18.50, 1.40),
            arrowprops=dict(arrowstyle="->", lw=1.4, color=C_ALERT))

# ═══════════════════════════════════════════════════════════════════════
# Legend (bottom-left)
# ═══════════════════════════════════════════════════════════════════════
leg = FancyBboxPatch((0.4, -0.15), 13.4, 0.60,
                     boxstyle="round,pad=0.03,rounding_size=0.05",
                     fc="#F8F8F8", ec="#888", lw=0.5)
ax.add_patch(leg)
ax.text(0.7, 0.15, "图例说明:", fontsize=8.5, weight="bold",
        ha="left", va="center", color="#333")

# Solid black arrow — data flow
ax.annotate("", xy=(3.0, 0.15), xytext=(2.4, 0.15),
            arrowprops=dict(arrowstyle="->", lw=0.9, color="#222"))
ax.text(3.10, 0.15, "数据流", fontsize=7.8, ha="left", va="center", color="#333")

# Dashed green — fast path
ax.plot([4.5, 5.1], [0.15, 0.15], color=C_EDGE_EC, lw=1.0,
        linestyle=(0, (4, 2)))
ax.annotate("", xy=(5.1, 0.15), xytext=(5.0, 0.15),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C_EDGE_EC))
ax.text(5.20, 0.15, "高置信度快速路径(Fast Path)",
        fontsize=7.8, ha="left", va="center", color="#333")

# Dashed blue — federated
ax.plot([9.5, 10.1], [0.15, 0.15], color=C_CLOUD_EC, lw=1.0,
        linestyle=(0, (4, 2)))
ax.annotate("", xy=(10.1, 0.15), xytext=(10.0, 0.15),
            arrowprops=dict(arrowstyle="->", lw=0.9, color=C_CLOUD_EC))
ax.text(10.20, 0.15, "跨用户知识聚合(无原始数据)",
        fontsize=7.8, ha="left", va="center", color="#333")

# Tier color legend (right side of legend bar)
ax.add_patch(Rectangle((14.10, 0.05), 0.4, 0.22, fc="white",
                       ec=C_EDGE_EC, lw=1.0))
ax.text(14.65, 0.15, "端侧(Edge)", fontsize=7.8, ha="left", va="center",
        color="#333")
ax.add_patch(Rectangle((16.65, 0.05), 0.4, 0.22, fc="white",
                       ec=C_CLOUD_EC, lw=1.0))
ax.text(17.20, 0.15, "云端(Cloud)", fontsize=7.8, ha="left", va="center",
        color="#333")
ax.add_patch(Rectangle((19.30, 0.05), 0.4, 0.22, fc="white",
                       ec=C_FUSION_EC, lw=1.0))
ax.text(19.85, 0.15, "融合与决策", fontsize=7.8, ha="left", va="center",
        color="#333")

# Abbreviation glossary (bottom-right)
gloss = FancyBboxPatch((13.85, -0.15), 7.75, 0.60,
                       boxstyle="round,pad=0.03,rounding_size=0.05",
                       fc="#F8F8F8", ec="#888", lw=0.5)
# Place glossary BELOW the legend instead — adjust position
# Actually use a single bottom row with combined content
# Remove this duplicate gloss patch (legend already takes the row)

# Add abbreviation row below legend
ax.text(0.7, -0.55,
        "缩略词:DP = Differential Privacy(差分隐私);  CoT = Chain-of-Thought(链式推理);  "
        "NVFP4 = 4-bit NormalFloat;  Q4_K_M = 4-bit 量化格式",
        fontsize=7, ha="left", va="center", color="#444", style="italic")

# Save
plt.tight_layout(pad=0.3)
sci.save(fig, "fig01_architecture.png", w=11.0, h=8.0)
print("Saved fig01_architecture.png")
