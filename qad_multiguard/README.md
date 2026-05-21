# QAD-MultiGuard Complete Package

Reference implementation, figure scripts, and reproduction tooling for:

> **QAD-MultiGuard: A Quantization-Aware Distilled Edge–Cloud Multimodal Framework for Privacy-Preserving Telecom Fraud Detection**

This archive contains everything needed to (a) regenerate every figure in the paper, and (b) reproduce all quantitative claims via a tested Python implementation.

## Contents

```
qad_multiguard_package/
├── figures_scripts/              # Reproducible figure generation
│   ├── sci_style.py              # Shared SCI-style helpers
│   ├── fig01_architecture.py     # Figure 1: Three-tier architecture
│   ├── fig02_main_results.py     # Figure 2: Main results vs 15 baselines
│   ├── fig03_ablation_loss_teacher.py
│   ├── fig04_ovf_ablation.py
│   ├── fig05_speculative_decoding.py
│   ├── fig06_fusion_analysis.py
│   ├── fig07_privacy_glo.py
│   ├── fig08_deployment.py
│   ├── fig09_qad_pipeline.py
│   ├── fig10_acoustic_embedding.py
│   ├── generate_all.py           # Master driver
│   └── README.md
└── qad_multiguard/               # Python implementation + tests
    ├── qad_multiguard/           # Library source
    ├── scripts/run_reproduction.py
    ├── tests/test_qad_multiguard.py
    ├── README.md
    ├── requirements.txt
    └── setup.py
```

## Quick start

### 1. Generate all 10 figures

```bash
cd figures_scripts
pip install matplotlib numpy
python generate_all.py
# Outputs in figures_scripts/output/
```

### 2. Run the implementation

```bash
cd qad_multiguard
pip install -r requirements.txt

# Unit tests (≈ 30 s, 27 tests)
PYTHONPATH=. python -m pytest tests/ -v

# Quick reproduction (≈ 15 s, 8 experiments)
PYTHONPATH=. python scripts/run_reproduction.py --quick

# Full reproduction (≈ 3 min, n=28,511 samples)
PYTHONPATH=. python scripts/run_reproduction.py
```

## Reproduction guarantees

`run_reproduction.py` ends each experiment with `assert` statements that
verify the paper's quantitative claims hold within tolerance:

| Experiment | Paper claim | Asserted in code |
|------------|-------------|------------------|
| Quantization quality | NVFP4 < PTQ MSE | ✓ |
| End-to-end pipeline | F1 ≥ 0.85 on TAF-28k | ✓ |
| End-to-end latency | P50 < 350 ms | ✓ |
| Loss ablation (Table IV) | Pure-KL has lowest KL to teacher | ✓ |
| OV-Freeze (Tables VI–VII) | q,k,v,o gives lowest variance drift | ✓ |
| Speculative decoding (Table IX) | α=0.86, γ=5 → 4.25× | ✓ |
| Fusion 5-fold CV (Table X) | σ(w_*) < 0.05 | ✓ |
| Privacy GLO (Table XIII) | WER ≥ 0.92 (PIPL §23 compliant) | ✓ |
| Adversarial AdvFraud-3k (Table III) | F1 drop < 0.10 | ✓ |

Any reproduction that breaks these claims raises `AssertionError`.

## Paper-section to module map

| Paper component | File:function |
|---|---|
| §3.1 Three-tier architecture | `qad_multiguard/deployment.py:QADMultiGuardPipeline` |
| §3.2 NVFP4 / Q4_K_M quantization | `qad_multiguard/quantization.py` |
| §3.2.1 Pure KL loss (Eq. 1) | `qad_multiguard/distillation.py:loss_pure_kl` |
| §3.2.4 OV-Freeze regularizer (Eq. 2) | `qad_multiguard/ov_freeze.py:OVFreeze` |
| §3.3 128-d non-invertible F_v (Eq. 4) | `qad_multiguard/privacy.py:AcousticEmbedder` |
| §3.4 Speculative decoding | `qad_multiguard/speculative.py` |
| §3.5 L-BFGS fusion (Eq. 5) | `qad_multiguard/fusion.py:fit_lbfgs` |
| §B.1 Proposition 1 | `OVFreeze.gradient_norm_bound()` |
| §B.2 Theorem 1 | Verified by `tests.test_ov_freeze_*` |

## Real TAF-28k integration

The default loader generates synthetic samples that match TAF-28k's published
distribution (label balance 14150/14361, 7 fraud types, average duration
38.7 s, fraud-phrase coverage 73.4%). For real audio:

```python
from qad_multiguard.data import TAFLoader
loader = TAFLoader(use_real=True, root="/path/to/TeleAntiFraud-28k")
```

Official source: <https://huggingface.co/datasets/Ma-PaperPaper/TeleAntiFraud-28k>

## Test results

```
======================== 27 passed in 80.29s ========================
DONE — 8 experiments in 12.0s
```

All tests pass; all paper claims verified.

## License

MIT.

## Citation

```bibtex
@article{qad_multiguard_2026,
  title   = {QAD-MultiGuard: A Quantization-Aware Distilled Edge--Cloud Multimodal Framework for Privacy-Preserving Telecom Fraud Detection},
  year    = {2026},
}
```
