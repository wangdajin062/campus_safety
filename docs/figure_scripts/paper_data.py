"""
paper_data.py - Single source of truth for ALL QAD-MultiGuard paper figures.

Every number below has been cross-checked against the paper's tables and body
text (Table 4 main results, Table 3 fusion, Table 6 cross-dataset, Table 8
speculative decoding, Table 9 privacy) and the engineering-deployment
experiment runs (EXP01-EXP10). Recovery rates satisfy
``recovery = F1 / BF16_F1 * 100`` exactly, and the latency components sum to
the paper's P50 = 268 ms / P99 = 342 ms.

Audit status (cross-checked): all figure-bearing numbers CONSISTENT with the
paper. The only orphan in the original codebase was a completed-deployment
metric block (5,000-student pilot) that the paper deliberately frames as a
*planned* 2,000-user pilot; that block is NOT used by any of the seven paper
figures and is therefore excluded here to avoid a status mismatch.
"""

# --- Project-wide constants ------------------------------------------------
BF16_F1      = 0.931
BF16_F1_ERR  = 0.005

NVFP4_SIZE_MB  = 248
Q4_K_M_SIZE_MB = 240

SAFE_QAQ_F1     = 0.918
SAFE_QAQ_F1_ERR = 0.006

# --- Table 4 / Figure 3 : main results (TAF-28k) ---------------------------
# recovery = F1 / 0.931 * 100 (verified to 1 d.p.)
EXP01_QUANT_QUALITY = [
    {"key": "ptq_baseline", "name": "Plain RTN PTQ",     "f1": 0.838, "recovery": 90.0, "std": 0.011},
    {"key": "awq",          "name": "NVFP4 + AWQ",       "f1": 0.838, "recovery": 90.0, "std": 0.010},
    {"key": "gptq",         "name": "NVFP4 + GPTQ",      "f1": 0.840, "recovery": 90.2, "std": 0.010},
    {"key": "spinquant",    "name": "NVFP4 + SpinQuant", "f1": 0.838, "recovery": 90.0, "std": 0.011},
    {"key": "quarot",       "name": "NVFP4 + QuaRot",    "f1": 0.838, "recovery": 90.0, "std": 0.011},
    {"key": "bitdistiller", "name": "NVFP4 + BitDistill","f1": 0.858, "recovery": 92.2, "std": 0.009},
]
QAT_QAD_OVF = [
    {"name": "NVFP4 QAT (CE)",         "f1": 0.844, "f1_err": 0.014, "recovery": 90.7},
    {"name": "NVFP4 QAD",              "f1": 0.916, "f1_err": 0.007, "recovery": 98.4},
    {"name": "NVFP4 QAD + OV-Freeze",  "f1": 0.923, "f1_err": 0.006, "recovery": 99.1},
    {"name": "Q4_K_M QAD + OV-Freeze", "f1": 0.917, "f1_err": 0.007, "recovery": 98.5},
]

# --- §3.3.1 latency decomposition (P50=268, P99=342) -----------------------
LATENCY_COMPONENTS = ["Feat.", "Fast", "CoT spec.", "Fus.+UI"]
LATENCY_P50_MS = [16, 28, 212, 12]   # sum = 268
LATENCY_P99_MS = [22, 36, 268, 16]   # sum = 342

# --- Figure 4 : loss-convergence trace (illustrative deterministic curve) --
# Headline numbers (paper caption): plateau ~0.045 -> 0.016 after OVF at
# step 1,400 (2.76x drop); SNR stable 18.4-18.9 dB.
LOSS_PLATEAU = 0.045
LOSS_CONVERGED = 0.016
OVF_ACTIVATION_STEP = 1400
TOTAL_STEPS = 2000
SNR_RANGE = (18.4, 18.9)

# --- Figure 5(a) / EXP03 : loss-function ablation --------------------------
EXP03_LOSS_ABLATION = [
    {"loss": "Pure KL\n(ours)", "f1": 0.916, "kl": 0.005, "std": 0.007},
    {"loss": "MSE",            "f1": 0.901, "kl": 0.082, "std": 0.010},
    {"loss": "CE\n(= QAT)",    "f1": 0.844, "kl": 0.311, "std": 0.014},
    {"loss": "3-term\nhybrid", "f1": 0.879, "kl": 0.124, "std": 0.012},
    {"loss": "KL +\ntask",     "f1": 0.908, "kl": 0.041, "std": 0.009},
]

# --- Figure 5(b) / EXP09 : teacher selection -------------------------------
EXP09_TEACHER = [
    {"teacher": "0.5B\n(same)", "f1_fixed": 0.916, "f1_conv": 0.916, "tokens_B": 0.5},
    {"teacher": "1.8B",         "f1_fixed": 0.911, "f1_conv": 0.913, "tokens_B": 0.7},
    {"teacher": "3B",           "f1_fixed": 0.904, "f1_conv": 0.910, "tokens_B": 1.0},
    {"teacher": "7B",           "f1_fixed": 0.892, "f1_conv": 0.915, "tokens_B": 2.0},
]

# --- Figure 6(a) / EXP04 : OV-Freeze layer-selection ablation --------------
EXP04_OVF_LAYER_ABLATION = [
    {"config": "no OVF",      "f1": 0.916, "drift_pct": 18.2},
    {"config": "FFN",         "f1": 0.918, "drift_pct": 15.4},
    {"config": "q",           "f1": 0.918, "drift_pct":  9.4},
    {"config": "q,v",         "f1": 0.920, "drift_pct":  5.1},
    {"config": "q,k,v",       "f1": 0.922, "drift_pct":  2.8},
    {"config": "q,k,v,o\n(ours)", "f1": 0.923, "drift_pct": 1.3},
    {"config": "q,k,v,o\n+FFN",   "f1": 0.922, "drift_pct": 1.5},
]

# --- Figure 6(b) / EXP10 : OV-Freeze activation step-ratio -----------------
EXP10_OVF_STEP_RATIO = [
    {"ratio_pct":  0, "f1": 0.916, "ppl": 8.73},
    {"ratio_pct": 10, "f1": 0.919, "ppl": 8.68},
    {"ratio_pct": 20, "f1": 0.921, "ppl": 8.65},
    {"ratio_pct": 30, "f1": 0.923, "ppl": 8.62},
    {"ratio_pct": 40, "f1": 0.922, "ppl": 8.63},
    {"ratio_pct": 50, "f1": 0.918, "ppl": 8.66},
]

# --- Figure 7 / Table 8 / EXP05 : speculative decoding ---------------------
# Measured rows match Table 8 exactly (domain-tuned alpha=0.86):
#   g=5 -> H100 3.49 / SD8G3 3.32 ; g=7 -> 4.10/3.90 ; g=10 -> 4.74/4.51
EXP05_SPECULATIVE = {
    0.78: [  # generic draft
        {"gamma": 3,  "h100": 2.37, "sd8g3": 2.26},
        {"gamma": 5,  "h100": 2.92, "sd8g3": 2.78},
        {"gamma": 7,  "h100": 3.25, "sd8g3": 3.10},
        {"gamma": 10, "h100": 3.52, "sd8g3": 3.35},
    ],
    0.86: [  # anti-fraud domain-tuned
        {"gamma": 3,  "h100": 2.65, "sd8g3": 2.52},
        {"gamma": 5,  "h100": 3.49, "sd8g3": 3.32},
        {"gamma": 7,  "h100": 4.10, "sd8g3": 3.90},
        {"gamma": 10, "h100": 4.74, "sd8g3": 4.51},
    ],
}
SPEC_ALPHA_GENERIC = 0.78
SPEC_ALPHA_TUNED   = 0.86
SPEC_GAMMA_DEPLOY  = 5


def speedup(alpha: float, gamma: int) -> float:
    """Closed-form speculative-decoding speedup (Leviathan et al., 2023, Eq.1).

        Speedup = (1 - alpha^(gamma+1)) / (1 - alpha)
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    if alpha <= 0.0:
        return 1.0
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


if __name__ == "__main__":
    # Self-check: recovery consistency and speedup anchors.
    for m in EXP01_QUANT_QUALITY:
        assert abs(round(m["f1"] / BF16_F1 * 100, 1) - m["recovery"]) < 0.06, m
    for m in QAT_QAD_OVF:
        assert abs(round(m["f1"] / BF16_F1 * 100, 1) - m["recovery"]) < 0.06, m
    assert sum(LATENCY_P50_MS) == 268
    assert sum(LATENCY_P99_MS) == 342
    assert abs(speedup(0.78, 5) - 3.52) < 0.01
    assert abs(speedup(0.86, 5) - 4.25) < 0.01
    print("paper_data.py - all consistency self-checks pass")
