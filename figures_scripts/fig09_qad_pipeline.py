"""
Figure 9: QAD Training Pipeline
对应论文 §4.2: 纯 KL 散度蒸馏 + 同源教师 + OV-Freeze
设计: 清晰的两路并行流 → KL → 反向传播
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sci_style import PALETTE, apply_sci_style, save_fig

apply_sci_style()


def round_box(ax, x, y, w, h, title, sub, fc, ec, title_size=9, sub_size=7.5):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(box)
    if sub:
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=ec)
        ax.text(x + w/2, y + h*0.30, sub, ha='center', va='center',
                fontsize=sub_size, color='#404040', style='italic')
    else:
        ax.text(x + w/2, y + h/2, title, ha='center', va='center',
                fontsize=title_size, fontweight='bold', color=ec)


def arrow(ax, x1, y1, x2, y2, color='#404040', lw=1.2, curve=0, label=None,
          label_offset=(0, 0)):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->', color=color, linewidth=lw,
                        mutation_scale=14, connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)
    if label:
        lx = (x1 + x2) / 2 + label_offset[0]
        ly = (y1 + y2) / 2 + label_offset[1]
        ax.text(lx, ly, label, fontsize=7.5, color=color, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                           edgecolor='none', alpha=0.92))


def make_figure():
    fig, ax = plt.subplots(figsize=(13, 6.0))
    ax.set_xlim(-0.2, 14.2)
    ax.set_ylim(-0.2, 6.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.text(6.6, 6.20,
            'QAD Training Pipeline: Pure KL Distillation with Same-Source Teacher',
            fontsize=11.5, fontweight='bold', ha='center', color=PALETTE['navy'])

    # ── 输入 ─────────────────────────────────────────────
    round_box(ax, 0.3, 2.7, 1.5, 0.9, 'Input  x', 'anti-fraud\ndialogue',
              '#E8F1F8', PALETTE['gray'], title_size=10)
    ax.text(1.05, 2.3, 'shared input', fontsize=7.5, color=PALETTE['gray'],
            ha='center', style='italic')

    # ── 教师分支 (上路) ─────────────────────────────────
    round_box(ax, 2.5, 4.5, 2.4, 1.0, 'BF16 Teacher',
              'Qwen2.5-0.5B (frozen)', '#D6E4F0', PALETTE['blue'])
    round_box(ax, 5.4, 4.5, 1.8, 1.0, r'$p_T(y\,|\,x)$',
              'softmax, T = 1', '#B5D2EC', PALETTE['navy'])

    # ── 学生分支 (下路) ─────────────────────────────────
    round_box(ax, 2.5, 0.85, 2.4, 1.0, 'NVFP4 Student',
              'Qwen2.5-0.5B (trainable)', '#FCE4EC', PALETTE['red'])
    round_box(ax, 5.4, 0.85, 1.8, 1.0, r'$p_S(y\,|\,x)$',
              'softmax, T = 1', '#F5B7B1', PALETTE['crimson'])

    # ── KL 损失中心节点 ──────────────────────────────────
    round_box(ax, 7.8, 2.7, 2.6, 1.0, r'$L_{QAD} = D_{KL}(p_T \| p_S)$',
              'Pure KL — no task / MSE',
              '#FFF4E6', PALETTE['orange'], title_size=10)

    # ── 反向传播 (右侧大箭头框) ─────────────────────────
    round_box(ax, 11.0, 2.7, 2.5, 1.0,
              'Gradient Update', 'AdamW, η = 1e-5', '#E0F7FA', '#00838F',
              title_size=9.5)

    # ── OV-Freeze 模块（独立侧栏标注） ───────────────────
    ov_box = FancyBboxPatch((11.0, 0.85), 2.5, 1.0,
                             boxstyle="round,pad=0.02,rounding_size=0.05",
                             facecolor='#FCE4EC', edgecolor=PALETTE['red'],
                             linewidth=1.4, linestyle='--')
    ax.add_patch(ov_box)
    ax.text(12.25, 1.55, 'OV-Freeze (eq. 2)', ha='center', va='center',
            fontsize=9.5, fontweight='bold', color=PALETTE['red'])
    ax.text(12.25, 1.20, 'final 30% of steps · q,k,v,o-proj',
            ha='center', va='center', fontsize=7.5, color='#404040', style='italic')

    # ── 数据流箭头 ───────────────────────────────────────
    # 输入 → 教师 / 学生
    arrow(ax, 1.8, 3.4, 2.5, 5.0, color=PALETTE['gray'], lw=1.0, curve=0.15)
    arrow(ax, 1.8, 2.9, 2.5, 1.35, color=PALETTE['gray'], lw=1.0, curve=-0.15)

    # 教师内部
    arrow(ax, 4.9, 5.0, 5.4, 5.0, color=PALETTE['blue'], lw=1.4)
    # 学生内部
    arrow(ax, 4.9, 1.35, 5.4, 1.35, color=PALETTE['red'], lw=1.4)

    # 教师 + 学生 → KL (汇聚)
    arrow(ax, 7.2, 5.0, 7.9, 3.6, color=PALETTE['navy'], lw=1.4, curve=-0.15,
          label='p_T', label_offset=(0.05, 0.1))
    arrow(ax, 7.2, 1.35, 7.9, 2.8, color=PALETTE['crimson'], lw=1.4, curve=0.15,
          label='p_S', label_offset=(0.05, -0.1))

    # KL → Gradient Update
    arrow(ax, 10.4, 3.2, 11.0, 3.2, color=PALETTE['orange'], lw=1.6,
          label=r'$\nabla$L_QAD', label_offset=(0, 0.30))

    # Gradient Update → Student (return arrow, curving downward)
    arrow(ax, 12.25, 2.7, 12.25, 1.85, color='#00838F', lw=1.4)
    ax.text(13.20, 2.30, 'updates\nstudent\nweights', fontsize=7.5,
            color='#00838F', style='italic', ha='center', va='center')

    # OV-Freeze → 学生（左下方箭头）
    arrow(ax, 11.0, 1.35, 7.2, 1.10, color=PALETTE['red'], lw=1.0, curve=0.10)
    ax.text(9.0, 0.55, 'variance correction\nof q,k,v,o-proj', fontsize=7,
            color=PALETTE['red'], style='italic', ha='center')

    # ── 教师/学生分支大标签 ────────────────────────────
    ax.text(0.3, 5.7, 'Teacher branch', fontsize=10, fontweight='bold',
            color=PALETTE['blue'], style='italic')
    ax.text(0.3, 0.85, 'Student branch (trained by QAD)', fontsize=10,
            fontweight='bold', color=PALETTE['red'], style='italic')

    # ── Key properties side-bar (右下角) ──────────────
    props_box = FancyBboxPatch((0.3, 0.1), 6.0, 0.55,
                               boxstyle="round,pad=0.02,rounding_size=0.04",
                               facecolor='#F8F9FA', edgecolor=PALETTE['darkgray'],
                               linewidth=0.6, alpha=0.7)
    ax.add_patch(props_box)
    props = '• 0.5B tokens (~1.7% of original SFT data)    ' + \
            '• T = 1 (no soft-label smoothing)    ' + \
            '• Same-source teacher    ' + \
            '• 99.1% recovery vs BF16'
    ax.text(3.3, 0.38, props, fontsize=7.5, color=PALETTE['darkgray'],
            ha='center', va='center')

    # 标题脚注
    fig.text(0.5, 0.005,
             'Fig. 9.  The QAD pipeline. The BF16 teacher and the quantized student receive the same input; pure KL divergence flows back only to the student. OV-Freeze applies in the final 30% of training steps to the q/k/v/o projections.',
             ha='center', fontsize=9, style='italic', color='#404040')

    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_fig(fig, 'fig09_qad_pipeline',
             formats=('pdf', 'png'), out_dir='./output')
    plt.close(fig)
    print('Figure 9 done.')
