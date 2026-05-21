"""
sci_style.py — QAD-MultiGuard 论文配图统一 SCI 风格
─────────────────────────────────────────────────────────────
设计原则:
  1. 色系: SCI 顶刊蓝灰系 + 强调色（Nature/Cell/IEEE 风格）
  2. 字体: sans-serif 英文 / Noto Sans CJK 中文，统一字号
  3. 分辨率: 600 DPI；矢量优先 (PDF/SVG)
  4. 自明性: 标题、图例、数据标注、坐标轴标签齐全
  5. 无遮挡: 留白≥10%，图例置于不遮数据处
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    'navy':       '#1F3864',
    'blue':       '#2E5C8A',
    'sky':        '#4A90C2',
    'lightblue':  '#A4C8E1',
    'red':        '#C00000',
    'crimson':    '#8B0000',
    'orange':     '#E67E22',
    'amber':      '#F39C12',
    'gray':       '#595959',
    'darkgray':   '#404040',
    'lightgray':  '#BFBFBF',
    'background': '#F8F9FA',
    # 类别色
    'cat1': '#1F3864',
    'cat2': '#5B9BD5',
    'cat3': '#70AD47',
    'cat4': '#FFC000',
    'cat5': '#C00000',
    'cat6': '#7030A0',
    # 序数色
    'gradient_blue': ['#E8F1F8', '#B5D2EC', '#7FAEDC', '#4A90C2', '#2E5C8A', '#1F3864'],
    'gradient_red':  ['#FCE8E6', '#F5B7B1', '#E74C3C', '#C0392B', '#922B21', '#641E16'],
}

FONT_CONFIG = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 11,
    'mathtext.fontset': 'cm',
    'mathtext.default': 'regular',
}


def apply_sci_style():
    mpl.rcParams.update(FONT_CONFIG)
    mpl.rcParams.update({
        'figure.dpi': 100,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'none',
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#404040',
        'axes.labelcolor': '#000000',
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': '#E0E0E0',
        'grid.linewidth': 0.4,
        'grid.alpha': 0.7,
        'xtick.color': '#404040',
        'ytick.color': '#404040',
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#404040',
        'legend.fancybox': False,
        'legend.borderpad': 0.5,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'patch.linewidth': 0.8,
        'errorbar.capsize': 3,
    })


def add_panel_label(ax, label, x=-0.18, y=1.05, fontsize=12):
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left')


def save_fig(fig, name, formats=('pdf', 'png'), out_dir='.'):
    import os
    os.makedirs(out_dir, exist_ok=True)
    for fmt in formats:
        path = os.path.join(out_dir, f'{name}.{fmt}')
        # PDF: vector at 600 DPI; PNG: moderate 300 DPI for preview
        dpi = 600 if fmt == 'pdf' else 300
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=dpi)
        print(f'  ✓ saved {path}')


def use_chinese_font():
    """启用中文字体"""
    import matplotlib.font_manager as fm
    candidates = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'Noto Serif CJK SC',
                  'Source Han Sans CN', 'SimSun', 'Microsoft YaHei',
                  'WenQuanYi Zen Hei']
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)
    if chosen:
        mpl.rcParams['font.family'] = ['serif']
        mpl.rcParams['font.serif'] = [chosen, 'Times New Roman', 'Times', 'DejaVu Serif']
        mpl.rcParams['axes.unicode_minus'] = False
        print(f'  Using Chinese font: {chosen}')
    else:
        print('  Warning: no Chinese font found')
    return chosen
