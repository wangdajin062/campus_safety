"""
Evaluation metrics.

  * F1, Precision, Recall, FPR
  * Recovery rate (Acc(quantized) / Acc(BF16) × 100%)
  * KL divergence vs teacher
  * Wilcoxon signed-rank test
  * Cohen's kappa for inter-annotator agreement
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ClassificationMetrics:
    f1: float
    precision: float
    recall: float
    fpr: float
    accuracy: float
    n_pos: int
    n_neg: int


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Binary F1 / precision / recall / FPR. Labels in {0, 1}."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fpr = fp / max(1, fp + tn)
    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    return ClassificationMetrics(
        f1=f1, precision=precision, recall=recall, fpr=fpr,
        accuracy=accuracy, n_pos=int((y_true == 1).sum()),
        n_neg=int((y_true == 0).sum()),
    )


def recovery_rate(quantized_acc: float, bf16_acc: float) -> float:
    """Recovery rate (%) = Acc(quantized) / Acc(BF16) × 100."""
    if bf16_acc <= 0:
        return 0.0
    return 100.0 * quantized_acc / bf16_acc


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q). Both should be probability distributions."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float((p * np.log(p / q)).sum())


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired Wilcoxon signed-rank test. Returns p-value (approximate normal)."""
    diffs = np.asarray(x) - np.asarray(y)
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return {"statistic": 0.0, "p_value": 1.0, "n": 0}
    ranks = np.argsort(np.argsort(np.abs(diffs))) + 1
    signs = np.sign(diffs)
    w_plus = ranks[signs > 0].sum()
    w_minus = ranks[signs < 0].sum()
    w = min(w_plus, w_minus)
    n = len(diffs)
    # Normal approximation
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    z = (w - mean) / math.sqrt(var)
    # Two-sided p
    from math import erf
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / math.sqrt(2))))
    return {"statistic": float(w), "p_value": float(p), "z": float(z), "n": n}


def cohens_kappa(annotator_a: np.ndarray, annotator_b: np.ndarray, n_classes: int = 2) -> float:
    """Cohen's kappa for two annotators' categorical labels."""
    a = np.asarray(annotator_a).astype(int)
    b = np.asarray(annotator_b).astype(int)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for ai, bi in zip(a, b):
        cm[ai, bi] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = sum(cm[i, :].sum() * cm[:, i].sum() for i in range(n_classes)) / (n * n)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)
