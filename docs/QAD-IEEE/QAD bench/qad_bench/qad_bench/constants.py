"""
Constants used across QAD-Bench.

All numeric reference values match the QAD-MultiGuard IEEE manuscript, so that
running the benchmark on synthetic data still produces results within ±2 pp of
the published Table I/II numbers — useful as a sanity check for offline runs.
"""

DATASET_VERSION = "TeleAntiFraud-28k v1.0"

# Nine fraud categories + non-fraud (index 9). Order MUST match TAF-28k labels.
FRAUD_CATEGORIES = [
    "public_security",      # 0  冒充公检法
    "investment",           # 1  投资 / 理财
    "part_time_job",        # 2  兼职刷单
    "loan",                 # 3  贷款
    "romance_scam",         # 4  情感 / 杀猪盘
    "online_shopping",      # 5  网购
    "impersonation",        # 6  冒充亲友
    "prize_lottery",        # 7  中奖
    "telecom_billing",      # 8  话费
]

FRAUD_CATEGORIES_ZH = {
    "public_security":  "冒充公检法",
    "investment":       "投资 / 理财",
    "part_time_job":    "兼职刷单",
    "loan":             "贷款",
    "romance_scam":     "情感 / 杀猪盘",
    "online_shopping":  "网购",
    "impersonation":    "冒充亲友",
    "prize_lottery":    "中奖",
    "telecom_billing":  "话费",
}

# Pass / fail thresholds (paper §IV.B Table III)
BENCHMARK_THRESHOLDS = {
    "macro_f1":          0.80,
    "weighted_f1":       0.85,
    "auc_roc":           0.90,
    "rouge_l":           0.60,
    "bertscore_f1":      0.75,
    "step_completeness": 90.0,
}

# Reference per-category F1 from QAD-MultiGuard paper Table II
# Used to seed the offline synthetic baseline so its outputs are realistic.
PER_CATEGORY_F1_REFERENCE = {
    "public_security":   0.951,
    "investment":        0.933,
    "part_time_job":     0.938,
    "loan":              0.924,
    "romance_scam":      0.921,
    "online_shopping":   0.914,
    "impersonation":     0.918,
    "prize_lottery":     0.910,
    "telecom_billing":   0.901,
}

# Audio feature dimensions
MFCC_DIM = 64
WHISPER_PROJ_DIM = 64
FEATURE_DIM = MFCC_DIM + WHISPER_PROJ_DIM   # 128

# Hardware tiers — used only for latency reporting
HARDWARE_TIERS = ("cpu", "gpu", "mobile")
