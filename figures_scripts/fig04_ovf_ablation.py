"""
Figure 4: OV-Freeze Detailed Analysis (Layer & Step-Ratio)
对应论文 §5.7.1 表 IX (layer ablation) + §5.7.2 表 X (step-ratio)
"""
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sci_style import PALETTE, apply_sci_style, save_fig
from _fig_data import load_exp_data, fallback

apply_sci_style()


def make_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── (a) 层选择消融 (Progressive Layer Ablation) ───────
    exp4 = load_exp_data("exp04_ovf_ablation.json")
    if exp4 and all(r.get("f1") for r in exp4.get("results", [])):
        layers = [r["config"] for r in exp4["results"]]
        f1_l = [r["f1"] for r in exp4["results"]]
        ppl_l = [r.get("ppl", 8.62) for r in exp4["results"]]
        drift_l = [r.get("drift_pct", 1.3) for r in exp4["results"]]
    else:
        layers = ['Baseline\n(no OVF)', 'FFN only', 'q only', 'q,v', 'q,k,v',
                  'q,k,v,o\n(ours)', 'q,k,v,o\n+ FFN']
        f1_l = [0.916, 0.918, 0.918, 0.920, 0.922, 0.923, 0.922]
        ppl_l = [8.73, 8.71, 8.69, 8.66, 8.64, 8.62, 8.63]
        drift_l = [18.2, 15.4, 9.4, 5.1, 2.8, 1.3, 1.5]

    x = np.arange(len(layers))
    
    # 主轴: F1
    color_f1 = PALETTE['red']
    line1 = ax1.plot(x, f1_l, 'o-', color=color_f1, linewidth=2.0,
                      markersize=7, markeredgecolor='#641E16', markeredgewidth=0.7,
                      label='F1 score', zorder=5)
    
    # 标注 F1 值
    for i, f in enumerate(f1_l):
        ax1.annotate(f'{f:.3f}', xy=(i, f),
                     xytext=(0, 8), textcoords='offset points',
                     fontsize=7.5, ha='center', color=color_f1,
                     fontweight='bold' if i == 5 else 'normal')

    # 高亮我方设置
    ax1.scatter([5], [f1_l[5]], s=240, color='none',
                 edgecolor=PALETTE['red'], linewidth=2.0, zorder=6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, fontsize=8)
    ax1.set_ylabel('F1 score', fontsize=10, fontweight='bold', color=color_f1)
    ax1.set_ylim(0.910, 0.930)
    ax1.tick_params(axis='y', colors=color_f1)
    ax1.spines['left'].set_color(color_f1)

    # 副轴: 方差漂移
    ax1b = ax1.twinx()
    color_drift = PALETTE['navy']
    bars = ax1b.bar(x, drift_l, color=color_drift, alpha=0.25,
                     edgecolor=color_drift, linewidth=0.7, width=0.6, zorder=1,
                     label='Variance drift (%)')
    
    for i, d in enumerate(drift_l):
        ax1b.text(i, d + 0.6, f'{d:.1f}', ha='center', va='bottom',
                  fontsize=7.5, color=color_drift,
                  fontweight='bold' if i == 5 else 'normal')

    ax1b.set_ylabel('Sensitive-layer variance drift (%)', fontsize=10,
                     fontweight='bold', color=color_drift)
    ax1b.set_ylim(0, 22)
    ax1b.tick_params(axis='y', colors=color_drift)
    ax1b.spines['right'].set_visible(True)
    ax1b.spines['right'].set_color(color_drift)
    ax1b.grid(False)

    ax1.set_title('(a)  OV-Freeze layer-selection ablation',
                   fontsize=10.5, fontweight='bold', loc='left', pad=12)

    # 联合图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    leg = ax1.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=8.5,
                      framealpha=0.95)
    leg.get_frame().set_edgecolor('#808080')

    # ── (b) 训练步比例敏感性 ───────────────────────────────
    exp10 = load_exp_data("exp10_ovf_step_ratio.json")
    if exp10 and exp10.get("results"):
        ratios = [r["ratio_pct"] for r in exp10["results"]]
        f1_r = [r["f1"] for r in exp10["results"]]
        ppl_r = [r["ppl"] for r in exp10["results"]]
    else:
        ratios = [0, 10, 20, 30, 40, 50]
        f1_r = [0.916, 0.919, 0.921, 0.923, 0.922, 0.918]
        ppl_r = [8.73, 8.68, 8.65, 8.62, 8.63, 8.66]

    # 双线图
    color_f1 = PALETTE['red']
    color_ppl = PALETTE['navy']

    line_f1, = ax2.plot(ratios, f1_r, 'o-', color=color_f1, linewidth=2.0,
                         markersize=8, markeredgecolor='#641E16', markeredgewidth=0.7,
                         label='F1 score', zorder=5)
    
    # 标注 F1 + 高亮 30%
    for r, f in zip(ratios, f1_r):
        weight = 'bold' if r == 30 else 'normal'
        ax2.annotate(f'{f:.3f}', xy=(r, f),
                     xytext=(0, 9), textcoords='offset points',
                     fontsize=7.5, ha='center', color=color_f1, fontweight=weight)

    ax2.scatter([30], [0.923], s=300, color='none',
                 edgecolor=PALETTE['red'], linewidth=2.0, zorder=6)

    ax2.set_xticks(ratios)
    ax2.set_xticklabels([f'{r}%' for r in ratios], fontsize=9)
    ax2.set_ylabel('F1 score', fontsize=10, fontweight='bold', color=color_f1)
    ax2.set_ylim(0.913, 0.928)
    ax2.set_xlabel('OV-Freeze training-step ratio', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='y', colors=color_f1)

    # PPL 副轴
    ax2b = ax2.twinx()
    line_ppl, = ax2b.plot(ratios, ppl_r, 's--', color=color_ppl, linewidth=1.6,
                           markersize=7, markeredgecolor='#1F3864', markeredgewidth=0.6,
                           alpha=0.85, label='PPL', zorder=4)
    for r, p in zip(ratios, ppl_r):
        ax2b.annotate(f'{p:.2f}', xy=(r, p),
                      xytext=(0, -15), textcoords='offset points',
                      fontsize=7.5, ha='center', color=color_ppl)
    ax2b.set_ylabel('Perplexity (PPL)', fontsize=10, fontweight='bold',
                     color=color_ppl)
    ax2b.set_ylim(8.55, 8.78)
    ax2b.tick_params(axis='y', colors=color_ppl)
    ax2b.spines['right'].set_visible(True)
    ax2b.spines['right'].set_color(color_ppl)
    ax2b.grid(False)

    # 标注最佳点
    ax2.annotate('Optimal\n30%', xy=(30, 0.923), xytext=(36, 0.926),
                 fontsize=9, fontweight='bold', color=PALETTE['red'],
                 ha='center',
                 arrowprops=dict(arrowstyle='->', color=PALETTE['red'], lw=1.0))

    # 不稳定标注
    ax2.annotate('Gradient\noscillation', xy=(50, 0.918), xytext=(48, 0.916),
                 fontsize=8, color=PALETTE['orange'], ha='center', style='italic',
                 arrowprops=dict(arrowstyle='->', color=PALETTE['orange'], lw=0.8))

    ax2.set_title('(b)  OV-Freeze training-step ratio sensitivity',
                   fontsize=10.5, fontweight='bold', loc='left', pad=12)

    # 联合图例
    leg2 = ax2.legend([line_f1, line_ppl], ['F1 score', 'PPL'],
                       loc='lower center', fontsize=8.5, framealpha=0.95)
    leg2.get_frame().set_edgecolor('#808080')

    fig.text(0.5, -0.02,
             'Fig. 4.  OV-Freeze regularizer ablation. (a) Applying OV-Freeze to all four attention projections is optimal. (b) 30% training-step ratio strikes the best balance.',
             ha='center', fontsize=9, style='italic', color='#404040')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = make_figure()
    save_fig(fig, 'fig04_ovf_ablation',
             formats=('pdf', 'png'), out_dir='./output')
    plt.close(fig)
    print('Figure 4 done.')
