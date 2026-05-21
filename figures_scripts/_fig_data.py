"""
_fig_data.py -- Shared data bridge for figure scripts.

Loads experiment JSON from <repo_root>/runs/ and falls back to
hardcoded paper values when JSON files are unavailable.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


# runs/ is a sibling of figures_scripts/
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


# -- Paper fallback tables ------------------------------------------------
# Used when experiment JSON cannot be loaded.
# Structure: {figure_key: list_of_dicts}
PAPER_FALLBACK = {
    # -- fig02: Main results -- F1 comparison -----------------------------
    "fig02_methods": [
        ("BF16 (upper bound)",       0.931, 0.005, "darkgray"),
        ("NVFP4 PTQ (max)",          0.872, 0.009, "green"),
        ("NVFP4 + AWQ",              0.886, 0.008, "green"),
        ("NVFP4 + GPTQ",             0.891, 0.010, "green"),
        ("NVFP4 + QuaRot",           0.895, 0.009, "green"),
        ("NVFP4 + SpinQuant",        0.898, 0.007, "green"),
        ("NVFP4 + BitDistiller",     0.905, 0.011, "purple"),
        ("NVFP4 QAT",                0.844, 0.014, "gold"),
        ("NVFP4 QAD (ours)",         0.916, 0.007, "blue"),
        ("NVFP4 QAD + OVF (ours)",   0.923, 0.006, "red"),
        ("Q4_K_M PTQ",               0.851, 0.011, "lblue"),
        ("Q4_K_M QAD (ours)",        0.911, 0.008, "blue"),
        ("Q4_K_M QAD + OVF (ours)",  0.917, 0.007, "orange"),
        ("SAFE-QAQ [27]",            0.918, 0.006, "darkgray"),
        ("BERT-Fraud [14]",          0.876, 0.000, "darkgray"),
    ],

    # -- fig03: Loss ablation ---------------------------------------------
    "fig03_losses": [
        {"loss": "Pure KL\n(ours)",      "f1": 0.916, "kl": 0.007, "std": 0.005, "color": "red"},
        {"loss": "MSE",                  "f1": 0.901, "kl": 0.082, "std": 0.010, "color": "darkgray"},
        {"loss": "Cross\nEntropy",       "f1": 0.844, "kl": 0.311, "std": 0.014, "color": "darkgray"},
        {"loss": "Three-Term",           "f1": 0.879, "kl": 0.124, "std": 0.012, "color": "darkgray"},
        {"loss": "KL+Task\n+Reg",        "f1": 0.908, "kl": 0.041, "std": 0.009, "color": "darkgray"},
    ],

    # -- fig03b: Teacher selection ----------------------------------------
    "fig03_teachers": {
        "names": ['0.5B BF16\n(same-source,\nours)', '1.8B BF16', '3B BF16', '7B BF16'],
        "f1_fixed": [0.916, 0.911, 0.904, 0.892],
        "f1_converged": [0.916, 0.913, 0.910, 0.915],
        "tokens_conv": ["0.5B", "0.7B", "1.0B", "2.0B"],
    },

    # -- fig04a: OVF layer ablation ---------------------------------------
    "fig04_layers": {
        "names": ['Baseline\n(no OVF)', 'FFN only', 'q only', 'q,v', 'q,k,v',
                  'q,k,v,o\n(ours)', 'q,k,v,o\n+ FFN'],
        "f1": [0.916, 0.918, 0.918, 0.920, 0.922, 0.923, 0.922],
        "ppl": [8.73, 8.71, 8.69, 8.66, 8.64, 8.62, 8.63],
        "drift": [18.2, 15.4, 9.4, 5.1, 2.8, 1.3, 1.5],
    },

    # -- fig04b: OVF step ratio -------------------------------------------
    "fig04_step_ratio": {
        "ratios": [0, 10, 20, 30, 40, 50],
        "f1": [0.916, 0.919, 0.921, 0.923, 0.922, 0.918],
        "ppl": [8.73, 8.68, 8.65, 8.62, 8.63, 8.66],
    },

    # -- fig05: Speculative decoding measured speedups --------------------
    "fig05_speedups": {
        "gammas": [3, 5, 7, 10],
        "h100": [2.65, 3.49, 4.10, 4.74],
        "sd8g3": [2.52, 3.32, 3.90, 4.51],
        "kv_cache": [120, 200, 280, 400],
    },

    # -- fig06a: Progressive modality contribution ------------------------
    "fig06_progressive": {
        "modalities": ['Text\n(SMS)', '+ Metadata\n(call)', '+ URL\nfeatures', '+ Acoustic\n(128-d $F_v$)'],
        "f1": [0.872, 0.889, 0.901, 0.923],
        "deltas": ['---', '+0.017', '+0.012', '+0.022'],
    },

    # -- fig06b: 5-fold CV weights ----------------------------------------
    "fig06_weights": {
        "w_text": [0.41, 0.39, 0.40, 0.42, 0.38],
        "w_audio": [0.30, 0.31, 0.29, 0.30, 0.30],
        "w_url": [0.19, 0.20, 0.21, 0.19, 0.21],
        "w_meta": [0.10, 0.10, 0.10, 0.09, 0.11],
    },

    # -- fig06c: Architecture comparison ----------------------------------
    "fig06_archs": [
        {"arch": 'sigmoid\nlinear\n(ours)', "f1": 0.923, "latency": "<1 ms", "params": "5"},
        {"arch": 'softmax\nlinear',          "f1": 0.909, "latency": "<1 ms", "params": "5"},
        {"arch": 'MM-Transformer\n(2 layers)', "f1": 0.926, "latency": "8.2 ms", "params": "1.2M"},
        {"arch": 'MM-Transformer\n(4 layers)', "f1": 0.927, "latency": "16.4 ms", "params": "2.4M"},
    ],

    # -- fig07: Privacy GLO -----------------------------------------------
    "fig07_radar": {
        "categories": ['WER', '1-PESQ/5', '1-MOS/5', '1-SpkID', '1-MI'],
        "white_box": [0.95, 1 - 0.242, 1 - 0.236, 0.917, 1.0],
        "black_box": [0.97, 1 - 0.232, 1 - 0.222, 0.921, 1.0],
    },

    # -- fig08a: Latency breakdown ----------------------------------------
    "fig08_latency": {
        "feature": [18, 24],
        "fast": [32, 41],
        "cot": [218, 277],
        "agg": [12, 18],
        "totals": [280, 360],
    },

    # -- fig08b: Deployment metrics ---------------------------------------
    "fig08_deployment": {
        "metrics": ['Precision', 'Recall', 'User Satisfaction\n(scaled to %)'],
        "values": [93.2, 98.8, 92.0],
        "errors_low": [1.5, 1.4, 0],
        "errors_high": [1.3, 0.6, 0],
    },

    # -- fig08c: Head-to-head comparison ----------------------------------
    "fig08_head_to_head": {
        "categories": ['Bench\nAccuracy', 'Latency\n(median)', 'Model Size', 'PIPL §23\nCompliance'],
        "ours": [0.923, 1/268, 1/248, 1.0],
        "saqaq": [0.918, 1/1320, 1/7000, 0.5],
        "raw_ours": ['0.923', '268 ms', '248 MB', 'Full'],
        "raw_saqaq": ['0.918', '1320 ms', '~7000 MB', 'Partial'],
    },
}


# -- Public API ------------------------------------------------------------
def load_exp_data(exp_name: str) -> dict | None:
    """Load experiment JSON from runs/ directory. Returns None if not found."""
    path = RUNS_DIR / exp_name
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def fallback(key: str):
    """Return a deep copy of paper fallback data for a figure key."""
    return deepcopy(PAPER_FALLBACK.get(key, []))
