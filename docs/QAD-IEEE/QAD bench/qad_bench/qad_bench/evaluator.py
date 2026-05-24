"""
Detection + Reasoning evaluators (paper §IV.B Table III protocol).

Robust to missing optional deps — every metric either computes correctly
or returns a `None` with a logged reason, never raises.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List

import numpy as np

from .features import extract_features

log = logging.getLogger(__name__)


# ── Detection metrics ────────────────────────────────────────────────────────
def evaluate_detection(
    model,
    dataset,
    *,
    batch_size: int = 32,
    progress: bool = True,
) -> Dict:
    """
    Compute macro-F1, weighted-F1, AUC-ROC, per-category F1, latency.
    """
    try:
        from sklearn.metrics import f1_score, roc_auc_score, classification_report
    except ImportError as exc:
        raise RuntimeError("Please install: pip install scikit-learn") from exc

    from .constants import FRAUD_CATEGORIES

    all_preds:  List[int]  = []
    all_labels: List[int]  = []
    all_probs:  List[np.ndarray] = []
    latencies:  List[float] = []

    n = len(dataset)
    log.info("[Detection] Evaluating %d samples (batch=%d)...", n, batch_size)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = dataset.select(range(start, end))

        feats   = np.stack([extract_features(s.__dict__ if hasattr(s, "__dict__") else s) for s in batch])
        labels  = np.array([int(s["category_id"]) for s in batch])
        scripts = [s.get("transcript", "") if hasattr(s, "get") else getattr(s, "transcript", "") for s in batch]

        t0 = time.perf_counter()
        probs = model.predict_proba(feats, transcripts=scripts)
        latency = (time.perf_counter() - t0) * 1000.0 / max(1, len(batch))
        latencies.append(latency)

        preds = np.argmax(probs, axis=1)
        all_preds.extend(int(p) for p in preds)
        all_labels.extend(int(l) for l in labels)
        all_probs.extend(probs)

        if progress and (start // batch_size) % max(1, (n // (batch_size * 5))) == 0:
            log.info("  Progress: %d/%d (latency %.2f ms/sample)", end, n, latency)

    all_preds_a  = np.array(all_preds)
    all_labels_a = np.array(all_labels)
    all_probs_a  = np.stack(all_probs)

    macro_f1    = f1_score(all_labels_a, all_preds_a, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels_a, all_preds_a, average="weighted", zero_division=0)

    # AUC over the 9 fraud classes (skip class 9 / non-fraud for label space match)
    auc = float("nan")
    try:
        if len(np.unique(all_labels_a)) >= 2:
            label_set = list(range(all_probs_a.shape[1]))
            auc = roc_auc_score(
                all_labels_a, all_probs_a,
                multi_class="ovr",
                labels=label_set,
                average="macro",
            )
    except Exception as exc:                       # noqa: BLE001
        log.warning("AUC computation failed: %s", exc)

    rep = classification_report(
        all_labels_a, all_preds_a,
        target_names=FRAUD_CATEGORIES + ["non_fraud"],
        labels=list(range(10)),
        output_dict=True, zero_division=0,
    )
    per_cat = {c: round(rep.get(c, {}).get("f1-score", 0.0), 4) for c in FRAUD_CATEGORIES}

    return {
        "macro_f1":         round(float(macro_f1), 4),
        "weighted_f1":      round(float(weighted_f1), 4),
        "auc_roc":          round(float(auc), 4) if auc == auc else None,  # NaN check
        "per_category_f1":  per_cat,
        "latency_p50_ms":   round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms":   round(float(np.percentile(latencies, 95)), 2),
        "n_samples":        int(len(all_preds)),
    }


# ── Reasoning quality metrics ────────────────────────────────────────────────
def _step_complete(text: str) -> bool:
    return all(any(m in text for m in markers) for markers in (
        ("步骤1", "Step 1", "第一步"),
        ("步骤2", "Step 2", "第二步"),
        ("步骤3", "Step 3", "第三步"),
    ))


def evaluate_reasoning(
    model,
    dataset,
    *,
    max_samples: int = 500,
    progress: bool = True,
) -> Dict:
    """
    Compute ROUGE-L, BERTScore F1 (optional), and step completeness on CoT.
    """
    try:
        from rouge_score import rouge_scorer
    except ImportError as exc:
        raise RuntimeError("Please install: pip install rouge-score") from exc

    n = min(max_samples, len(dataset))
    log.info("[Reasoning] Evaluating %d CoT samples ...", n)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_scores: List[float] = []
    hyps: List[str] = []
    refs: List[str] = []
    step_complete = 0

    sub = dataset.select(range(n))
    for i, sample in enumerate(sub):
        ref  = getattr(sample, "cot_annotation", "") or ""
        hyp  = model.generate_cot(
            transcript = getattr(sample, "transcript", "") or "",
            audio      = getattr(sample, "audio", {}),
        )
        rouge_scores.append(scorer.score(ref, hyp)["rougeL"].fmeasure)
        hyps.append(hyp)
        refs.append(ref)
        if _step_complete(hyp):
            step_complete += 1

        if progress and i and i % max(1, n // 5) == 0:
            log.info("  Progress: %d/%d", i, n)

    # BERTScore is optional (heavy); skip if unavailable
    bertscore_f1 = None
    try:
        from bert_score import score as bs
        _, _, F1 = bs(hyps, refs, lang="zh", verbose=False)
        bertscore_f1 = round(F1.mean().item(), 4)
    except ImportError:
        log.info("[Reasoning] bert-score not installed; skipping BERTScore.")
    except Exception as exc:                       # noqa: BLE001
        log.warning("[Reasoning] BERTScore failed: %s", exc)

    return {
        "rouge_l":             round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
        "bertscore_f1":        bertscore_f1,
        "step_completeness":   round(step_complete / n * 100, 1),
        "n_reasoning_samples": int(n),
    }
