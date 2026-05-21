"""
Figure 10: Acoustic Embedding F_v Construction & Non-Invertibility
对应论文 §4.3: 128-d 不可逆声学嵌入构造 + 非可逆机制图解
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
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


def arrow(ax, x1, y1, x2, y2, color='#404040', lw=1.2, curve=0):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='->', color=color, linewidth=lw,
                        mutation_scale=14, connectionstyle=f"arc3,rad={curve}")
    ax.add_patch(a)


def make_figure():
    fig = plt.figure(figsize=(13, 7.5))
    
    # 上半: 流程图 (占 55%)
    ax_top = fig.add_axes([0.03, 0.45, 0.94, 0.50])
    ax_top.set_xlim(0, 14)
    ax_top.set_ylim(0, 4)
    ax_top.set_aspect('equal')
    ax_top.axis('off')

    # 下半: MFCC 时频图 + 信号样例 (占 35%)
    ax_bot1 = fig.add_axes([0.06, 0.07, 0.27, 0.28])
    ax_bot2 = fig.add_axes([0.39, 0.07, 0.27, 0.28])
    ax_bot3 = fig.add_axes([0.72, 0.07, 0.25, 0.28])

    # ── 上半: 流程图 ─────────────────────────────────────
    
    # 输入: 原始音频
    round_box(ax_top, 0.3, 1.6, 1.8, 1.0, 'Raw Audio',
              'on-device only', '#FFEBEE', PALETTE['red'])
    
    # 端侧标志
    device = FancyBboxPatch((0.15, 1.4), 2.1, 1.4,
                             boxstyle="round,pad=0.04,rounding_size=0.05",
                             facecolor='none', edgecolor=PALETTE['red'],
                             linewidth=1.5, linestyle='--')
    ax_top.add_patch(device)
    ax_top.text(1.2, 0.95, 'On-device only\n(PIPL §23)', fontsize=8,
                 color=PALETTE['red'], fontweight='bold', ha='center', style='italic')

    # 分支 1: MFCC 路径
    round_box(ax_top, 3.0, 2.5, 2.0, 0.85,
              'MFCC Filterbank',
              '64 Mel filters · 25 ms n_fft', '#D6E4F0', PALETTE['blue'])
    round_box(ax_top, 5.5, 2.5, 2.0, 0.85,
              'Time Average',
              'destroys phoneme\nsequence', '#FFE0B2', PALETTE['orange'])
    round_box(ax_top, 8.0, 2.5, 1.5, 0.85,
              r'$f_{mfcc}$', r'$\in \mathbb{R}^{64}$', '#B5D2EC', PALETTE['navy'])

    # 分支 2: Whisper-tiny 路径
    round_box(ax_top, 3.0, 0.7, 2.0, 0.85,
              'Whisper-tiny',
              'Encoder (CLS pool)', '#D6E4F0', PALETTE['blue'])
    round_box(ax_top, 5.5, 0.7, 2.0, 0.85,
              r'Projection $W_{proj}$',
              r'$\mathbb{R}^{384} \to \mathbb{R}^{64}$', '#FFE0B2', PALETTE['orange'])
    round_box(ax_top, 8.0, 0.7, 1.5, 0.85,
              r'$W \cdot \bar{h}_w$', r'$\in \mathbb{R}^{64}$',
              '#B5D2EC', PALETTE['navy'])

    # 拼接 + 输出
    round_box(ax_top, 10.2, 1.5, 1.6, 1.0,
              'Concatenate', '128-d', '#FCE4EC', PALETTE['red'])
    round_box(ax_top, 12.3, 1.4, 1.5, 1.2,
              r'$F_v$', '128-d\nnon-invertible',
              '#FFCDD2', PALETTE['crimson'], title_size=12)

    # 流程箭头
    arrow(ax_top, 2.1, 2.2, 3.0, 2.9, color=PALETTE['gray'], lw=1.0, curve=0.1)
    arrow(ax_top, 2.1, 2.0, 3.0, 1.1, color=PALETTE['gray'], lw=1.0, curve=-0.1)
    arrow(ax_top, 5.0, 2.9, 5.5, 2.9, color=PALETTE['gray'], lw=1.0)
    arrow(ax_top, 7.5, 2.9, 8.0, 2.9, color=PALETTE['gray'], lw=1.0)
    arrow(ax_top, 5.0, 1.1, 5.5, 1.1, color=PALETTE['gray'], lw=1.0)
    arrow(ax_top, 7.5, 1.1, 8.0, 1.1, color=PALETTE['gray'], lw=1.0)

    arrow(ax_top, 9.5, 2.9, 10.4, 2.3, color=PALETTE['navy'], lw=1.2, curve=0.1)
    arrow(ax_top, 9.5, 1.1, 10.4, 1.7, color=PALETTE['navy'], lw=1.2, curve=-0.1)
    arrow(ax_top, 11.8, 2.0, 12.3, 2.0, color=PALETTE['red'], lw=1.5)

    # 不可逆性标注（在分支外侧标注，避免遮挡子标题）
    ax_top.text(6.5, 3.65, 'Non-invertible step #1: time averaging',
                 fontsize=8, color=PALETTE['orange'], fontweight='bold',
                 ha='center', style='italic')
    ax_top.text(6.5, 0.55, 'Non-invertible step #2: CLS pooling',
                 fontsize=8, color=PALETTE['orange'], fontweight='bold',
                 ha='center', style='italic')

    # 上传箭头到云端
    arrow(ax_top, 13.2, 2.6, 13.4, 3.5, color=PALETTE['cat3'], lw=1.5, curve=0.0)
    ax_top.text(13.4, 3.7, 'Upload\n(if needed)', fontsize=8,
                 color=PALETTE['cat3'], fontweight='bold', ha='center')

    # 公式 — 顶部居中
    formula_box = FancyBboxPatch((4.5, 0.05), 5.0, 0.40,
                                   boxstyle="round,pad=0.03,rounding_size=0.05",
                                   facecolor='#F8F9FA', edgecolor=PALETTE['darkgray'],
                                   linewidth=0.8)
    # Don't add formula box at bottom - put it in dedicated position
    # Actually move it just above figure 10's bottom row at y=-0.05 (it was at 0.05)
    # Better: keep it but move the non-inv step #2 label
    ax_top.add_patch(formula_box)
    ax_top.text(7.0, 0.25, r'$F_v = [\, f_{mfcc};\; W_{proj} \cdot \bar{h}_w \,] \in \mathbb{R}^{128}$',
                 fontsize=10, ha='center', va='center', color=PALETTE['darkgray'])

    # ── 下半: 时频图样例 ─────────────────────────────────
    
    # (a) 原始音频时频图
    np.random.seed(42)
    t = np.linspace(0, 3, 480)
    freq_bands = 64
    spec = np.zeros((freq_bands, len(t)))
    # 模拟一段语音的 mel 频谱图
    for i, t_val in enumerate(t):
        for f in range(freq_bands):
            base = np.exp(-((f - 20)**2) / 100) * (1 + 0.5 * np.sin(t_val * 5))
            base += np.exp(-((f - 35)**2) / 80) * (0.7 + 0.3 * np.cos(t_val * 8))
            spec[f, i] = base + 0.1 * np.random.randn()
    
    im1 = ax_bot1.imshow(spec, aspect='auto', origin='lower',
                          extent=[0, 3, 0, 8000], cmap='viridis')
    ax_bot1.set_xlabel('Time (s)', fontsize=8)
    ax_bot1.set_ylabel('Frequency (Hz)', fontsize=8)
    ax_bot1.set_title('(a) Mel spectrogram\n(input, contains all info)',
                       fontsize=8.5, fontweight='bold')
    ax_bot1.tick_params(labelsize=7)

    # (b) 时间平均后的 MFCC (1D 向量)
    mfcc_avg = spec.mean(axis=1)
    ax_bot2.bar(np.arange(64), mfcc_avg, color=PALETTE['orange'],
                 edgecolor='#404040', linewidth=0.3)
    ax_bot2.set_xlabel('MFCC coefficient index', fontsize=8)
    ax_bot2.set_ylabel('Magnitude', fontsize=8)
    ax_bot2.set_title('(b) Time-averaged MFCC\n(64-d, sequence destroyed)',
                       fontsize=8.5, fontweight='bold')
    ax_bot2.tick_params(labelsize=7)
    ax_bot2.text(32, mfcc_avg.max() * 0.95,
                  'Phoneme order\nirrecoverable',
                  fontsize=7.5, color=PALETTE['red'],
                  fontweight='bold', ha='center', style='italic',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=PALETTE['red'], linewidth=0.6))

    # (c) F_v 最终向量 (128-d)
    np.random.seed(7)
    fv = np.random.randn(128) * 0.4
    fv[:64] = mfcc_avg / mfcc_avg.max()
    fv[64:] = np.tanh(np.random.randn(64))

    colors_fv = [PALETTE['orange'] if i < 64 else PALETTE['blue'] for i in range(128)]
    ax_bot3.bar(np.arange(128), fv, color=colors_fv,
                 edgecolor='none', width=0.9)
    ax_bot3.set_xlabel('F_v dimension', fontsize=8)
    ax_bot3.set_ylabel('Value', fontsize=8)
    ax_bot3.set_title(r'(c) Final $F_v$ embedding (128-d)' + '\n(uploaded to cloud)',
                       fontsize=8.5, fontweight='bold')
    ax_bot3.tick_params(labelsize=7)
    ax_bot3.axvline(63.5, color='#404040', linestyle='--', linewidth=0.6)
    ax_bot3.text(31, fv.max() * 0.85, r'$f_{mfcc}$', fontsize=9,
                  color=PALETTE['orange'], fontweight='bold', ha='center')
    ax_bot3.text(95, fv.max() * 0.85, r'$W \cdot \bar{h}_w$', fontsize=9,
                  color=PALETTE['blue'], fontweight='bold', ha='center')

    fig.text(0.5, 0.005,
             'Fig. 10.  Construction of the 128-d non-invertible acoustic embedding F_v. (a) Raw mel spectrogram contains full temporal information. (b) Time-averaging destroys phoneme order. (c) The final F_v concatenates MFCC and Whisper-projected features.',
             ha='center', fontsize=9, style='italic', color='#404040')

    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_fig(fig, 'fig10_acoustic_embedding',
             formats=('pdf', 'png'), out_dir='./output')
    plt.close(fig)
    print('Figure 10 done.')
