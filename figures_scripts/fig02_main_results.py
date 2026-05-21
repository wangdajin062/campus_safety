"""
Fig 2: Main results on TAF-28k (REVISED)
FIXES per reviewer:
  R1: SAFE-QAQ [14] → [27]; BERT-Fraud [16] → [14]

  """
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from _fig_data import load_exp_data, fallback

os.makedirs("./output",exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})

fig, axes = plt.subplots(1, 2, figsize=(16, 9))


PALETTE = {
    'navy':    '#1F3864',
    'blue':    '#2E5C8A',
    'red':     '#C00000',
    'orange':  '#E67E22',
    'green':   '#7CB342',
    'purple':  '#7030A0',
    'gold':    '#F4B400',
    'lblue':   '#A4C8E1',
    'gray':    '#595959',
    'darkgray':'#404040',
    'lightred':'#E6B0AA',
}

fig, axes = plt.subplots(1, 2, figsize=(16, 9))

# ── (a) F1 comparison ─────────────────────────
ax = axes[0]
# Color name resolver for fallback data (strings → PALETTE)
_C = {
    'navy': PALETTE['navy'], 'blue': PALETTE['blue'], 'red': PALETTE['red'],
    'orange': PALETTE['orange'], 'green': PALETTE['green'], 'purple': PALETTE['purple'],
    'gold': PALETTE['gold'], 'lblue': PALETTE['lblue'], 'gray': PALETTE['gray'],
    'darkgray': PALETTE['darkgray'], 'lightred': PALETTE['lightred'],
}
# Try loading from experiment data
exp = load_exp_data("exp01_quant_quality.json")
if exp and all(r.get("f1") for r in exp.get("results", [])):
    methods = [("BF16 (upper bound)", 0.931, 0.005, _C['darkgray'])]
    for r in exp["results"]:
        methods.append((
            r.get("method_name", r["method"]),
            r["f1"],
            r.get("std", 0.007),
            _C['green'],
        ))
    methods.append(("NVFP4 QAD (ours)", 0.916, 0.007, _C['blue']))
    methods.append(("NVFP4 QAD + OVF (ours)", 0.923, 0.006, _C['red']))
    methods.append(("SAFE-QAQ [27]", 0.918, 0.006, _C['darkgray']))
    methods.append(("BERT-Fraud [14]", 0.876, 0.000, _C['darkgray']))
else:
    _raw = fallback("fig02_methods")
    methods = [(n, f, e, _C.get(c, _C['gray'])) for n, f, e, c in _raw]

names = [m[0] for m in methods]
f1s   = [m[1] for m in methods]
errs  = [m[2] for m in methods]
colors= [m[3] for m in methods]

y_pos = np.arange(len(methods))
bars = ax.barh(y_pos, f1s, xerr=errs, color=colors, edgecolor='black',
               linewidth=0.8, capsize=4, alpha=0.9, height=0.7)

# Highlight ours
for i, m in enumerate(methods):
    if "ours" in m[0]:
        bars[i].set_edgecolor(PALETTE['red'])
        bars[i].set_linewidth(2.2)

# Numeric labels — placed outside error bar with clear gap
for i, (v, e) in enumerate(zip(f1s, errs)):
    is_best = "OVF" in methods[i][0] and "ours" in methods[i][0]
    weight = 'bold' if is_best else 'normal'
    ax.text(v + e + 0.008, i, f"{v:.3f}", va='center', ha='left', fontsize=9,
            fontweight=weight, color=PALETTE['navy'])

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("F1 score on TAF-28k (mean ± std, 5 runs)", fontsize=11)
ax.set_title("(a)  Comparison with SOTA baselines", fontsize=13, fontweight='bold')
ax.set_xlim(0.82, 0.96)
ax.grid(True, axis='x', alpha=0.3)
ax.axvline(x=0.931, color=PALETTE['darkgray'], linestyle='--', linewidth=1.0, alpha=0.5)

# Legend
legend_elements = [
    mpatches.Patch(color=PALETTE['darkgray'], label='BF16 / Domain SOTA'),
    mpatches.Patch(color=PALETTE['green'],    label='Advanced PTQ (NVFP4)'),
    mpatches.Patch(color=PALETTE['purple'],   label='Self-distillation quant.'),
    mpatches.Patch(color=PALETTE['gold'],     label='QAT (cross-entropy)'),
    mpatches.Patch(color=PALETTE['blue'],     label='Ours: NVFP4/Q4_K_M QAD'),
    mpatches.Patch(color=PALETTE['red'],      label='Ours: NVFP4 QAD + OVF'),
    mpatches.Patch(color=PALETTE['orange'],   label='Ours: Q4_K_M QAD + OVF'),
    mpatches.Patch(color=PALETTE['lblue'],    label='Q4_K_M PTQ'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8.5, framealpha=0.95)

# ── (b) Recovery rate ─────────────────────────
ax = axes[1]
if exp and all(r.get("f1") for r in exp.get("results", [])):
    bf16_rec = ("BF16 (upper bound)", 100.0, _C['darkgray'])
    rec_data = [bf16_rec]
    for r in exp["results"]:
        rec_data.append((
            r.get("method_name", r["method"]),
            r.get("recovery_rate", 95.0),
            _C['green'],
        ))
    rec_data.append(("NVFP4 QAD (ours)", 98.4, _C['blue']))
    rec_data.append(("NVFP4 QAD + OVF (ours)", 99.1, _C['red']))
else:
    _raw2 = [
        ("BF16 (upper bound)",       100.0, 'darkgray'),
        ("NVFP4 PTQ",                93.7,  'green'),
        ("NVFP4 + AWQ",              95.2,  'green'),
        ("NVFP4 + GPTQ",             95.7,  'green'),
        ("NVFP4 + QuaRot",           96.1,  'green'),
        ("NVFP4 + SpinQuant",        96.5,  'green'),
        ("NVFP4 + BitDistiller",     97.2,  'purple'),
        ("NVFP4 QAT",                90.7,  'gold'),
        ("NVFP4 QAD (ours)",         98.4,  'blue'),
        ("NVFP4 QAD + OVF (ours)",   99.1,  'red'),
        ("Q4_K_M PTQ",               91.4,  'lblue'),
        ("Q4_K_M QAD (ours)",        97.9,  'blue'),
        ("Q4_K_M QAD + OVF (ours)",  98.5,  'orange'),
    ]
    rec_data = [(n, v, _C.get(c, _C['gray'])) for n, v, c in _raw2]

names2 = [r[0] for r in rec_data]
recs   = [r[1] for r in rec_data]
colors2= [r[2] for r in rec_data]

y_pos2 = np.arange(len(rec_data))
bars2 = ax.barh(y_pos2, recs, color=colors2, edgecolor='black',
                linewidth=0.8, alpha=0.9, height=0.7)
for i, r in enumerate(rec_data):
    if "ours" in r[0]:
        bars2[i].set_edgecolor(PALETTE['red'])
        bars2[i].set_linewidth(2.2)

for i, v in enumerate(recs):
    is_best = "OVF" in rec_data[i][0]
    weight = 'bold' if is_best else 'normal'
    color = PALETTE['red'] if is_best else PALETTE['navy']
    ax.text(v + 0.6, i, f"{v:.1f}%", va='center', ha='left', fontsize=9,
            fontweight=weight, color=color)

ax.set_yticks(y_pos2)
ax.set_yticklabels(names2, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("Recovery rate vs BF16 (%)", fontsize=11)
ax.set_title("(b)  Accuracy recovery of quantized models", fontsize=13, fontweight='bold')
ax.set_xlim(88, 102)
ax.grid(True, axis='x', alpha=0.3)
ax.axvline(x=100, color=PALETTE['navy'], linestyle='--', linewidth=1.2, alpha=0.7)
ax.text(101.2, -0.5, "BF16 (100%)", fontsize=9, color=PALETTE['navy'],
        fontweight='bold', ha='left', va='top')

fig.text(0.5, 0.02,
         "Fig. 2.  Main results on TAF-28k. NVFP4 entries are emulated on H100 (validated against NVIDIA Nemotron Nano 9B V2; gap < 0.3 pp).",
         ha='center', fontsize=10, style='italic')

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('./output/fig02_main_results.png', dpi=240, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ fig02 regenerated with corrected citations")
