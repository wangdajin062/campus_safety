"""
QAD-Bench v1.1 — Reproducible Telecom Fraud Detection Benchmark
================================================================

Tiered evaluation harness for the QAD-MultiGuard system on the TeleAntiFraud-28k
dataset, with offline fallbacks so the benchmark always runs to completion.

Public API
----------
    from qad_bench import (
        QADMultiGuardModel,        # reference end-to-end model
        load_dataset,              # tiered loader: HF → cache → synthetic
        evaluate_detection,
        evaluate_reasoning,
        run_benchmark,
        FRAUD_CATEGORIES,
        BENCHMARK_THRESHOLDS,
    )

    results = run_benchmark(model_path="auto", hardware="cpu")

Citation: Ma et al., ACM MM 2025; Wang et al., arXiv:2601.01392
"""

from .constants import (
    FRAUD_CATEGORIES,
    FRAUD_CATEGORIES_ZH,
    BENCHMARK_THRESHOLDS,
    PER_CATEGORY_F1_REFERENCE,
    DATASET_VERSION,
)
from .dataset import load_dataset, DatasetSource
from .features import extract_features, compute_mfcc, compute_whisper_embedding
from .model import QADMultiGuardModel, build_default_model
from .evaluator import evaluate_detection, evaluate_reasoning
from .runner import run_benchmark

__version__ = "1.1.0"
__all__ = [
    "QADMultiGuardModel",
    "build_default_model",
    "load_dataset",
    "DatasetSource",
    "extract_features",
    "compute_mfcc",
    "compute_whisper_embedding",
    "evaluate_detection",
    "evaluate_reasoning",
    "run_benchmark",
    "FRAUD_CATEGORIES",
    "FRAUD_CATEGORIES_ZH",
    "BENCHMARK_THRESHOLDS",
    "PER_CATEGORY_F1_REFERENCE",
    "DATASET_VERSION",
]
