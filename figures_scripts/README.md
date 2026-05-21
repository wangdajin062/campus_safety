# QAD-MultiGuard Figure Scripts (10 figures)

Source for all figures in the QAD-MultiGuard paper.

## Quick start

```bash
pip install matplotlib numpy
python generate_all.py            # render all 10 figures
python generate_all.py --fig 5    # render only Figure 5
```

Outputs land in `./output/` as PNG (and PDF for some).

## Files

| File | Figure | Description |
|------|--------|-------------|
| `sci_style.py` | (lib) | Shared SCI-style palette and helpers |
| `fig01_architecture.py` | Fig 1 | Three-tier edge–cloud architecture |
| `fig02_main_results.py` | Fig 2 | TAF-28k main results vs 15 baselines |
| `fig03_ablation_loss_teacher.py` | Fig 3 | Loss + teacher selection ablations |
| `fig04_ovf_ablation.py` | Fig 4 | OV-Freeze layer + step ratio ablations |
| `fig05_speculative_decoding.py` | Fig 5 | Speculative decoding speedup analysis |
| `fig06_fusion_analysis.py` | Fig 6 | Multimodal fusion analysis (3 panels) |
| `fig07_privacy_glo.py` | Fig 7 | GLO white-box + black-box privacy attacks |
| `fig08_deployment.py` | Fig 8 | 30-day IRB deployment results |
| `fig09_qad_pipeline.py` | Fig 9 | QAD training pipeline diagram |
| `fig10_acoustic_embedding.py` | Fig 10 | F_v 128-d non-invertible embedding |
| `generate_all.py` | (driver) | Master script |

## Numerical sources

All numbers in the figures correspond to tables in the paper. Specifically:

- Fig 2 ↔ Table II
- Fig 3 ↔ Tables IV (loss) and V (teacher)
- Fig 4 ↔ Tables VI (layer) and VII (step ratio)
- Fig 5 ↔ Table IX (speculative decoding)
- Fig 6 ↔ Tables X, XI, XII (fusion)
- Fig 7 ↔ Table XIII (GLO)
- Fig 8 ↔ §4.13 (deployment)

## Reviewer fixes applied (v4)

This version of the scripts incorporates 15 reviewer issues:

- Citation numbers match paper bibliography (SAFE-QAQ [27], BERT-Fraud [14])
- 5 loss functions in Fig 3(a), 4 teacher conditions × 2 stopping criteria in Fig 3(b)
- "<1 ms" label in Fig 6(c) matches Table XII
- Privacy radar axes use "1−PESQ↑/1−MOS↑" so all axes consistently mean "more privacy outward"
- Fig 8(c) uses 268 ms (measured median per §4.13), 4.9× speedup
- Theoretical speedups in Fig 5 use the correct (1−α^(γ+1))/(1−α) formula
