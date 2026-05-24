"""
Top-level benchmark runner — programmatic and CLI entry point.

Usage (programmatic):
    from qad_bench import run_benchmark
    results = run_benchmark(
        model_path="auto",          # or path to .gguf / .pt
        split="test",
        hardware="cpu",
        prefer_offline=True,        # skip HF probe if you know you're offline
        skip_reasoning=False,
    )

Usage (CLI):
    python -m qad_bench.runner --model_path auto --hardware cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

from .constants import BENCHMARK_THRESHOLDS, FRAUD_CATEGORIES, FRAUD_CATEGORIES_ZH
from .dataset import load_dataset, DatasetSource
from .evaluator import evaluate_detection, evaluate_reasoning
from .model import build_default_model

log = logging.getLogger("qad_bench")


# ── Pretty printer ────────────────────────────────────────────────────────────
def _bar(value: float, width: int = 20, max_val: float = 1.0) -> str:
    filled = int((value / max_val) * width) if max_val else 0
    return "█" * max(0, min(width, filled))


def print_results(results: Dict) -> None:
    det = results["detection"]
    rea = results.get("reasoning") or {}

    print("\n" + "=" * 72)
    print("  QAD-BENCH v1.1 — RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Model        : {results['model']}")
    print(f"  Mode         : {results['mode']}")
    print(f"  Hardware     : {results['hardware']}")
    print(f"  Dataset      : {results['dataset_source']} (split={results['split']}, n={det['n_samples']})")
    print(f"  Timestamp    : {results['timestamp']}")
    print()

    print("  ── DETECTION PERFORMANCE ────────────────────────────────────────────")
    for metric in ("macro_f1", "weighted_f1", "auc_roc"):
        if metric in det and det[metric] is not None:
            val = det[metric]
            thr = BENCHMARK_THRESHOLDS.get(metric, 0)
            ok  = "[PASS]" if val >= thr else "[FAIL]"
            print(f"  {metric:24s} : {val:.4f}   (threshold {thr:.2f})   {ok}")

    print()
    print("  ── PER-CATEGORY F1 ──────────────────────────────────────────────────")
    for cat in FRAUD_CATEGORIES:
        f1 = det["per_category_f1"].get(cat, 0.0)
        zh = FRAUD_CATEGORIES_ZH.get(cat, "")
        bar = _bar(f1, 18)
        print(f"  {zh:14s} ({cat:18s}) : {f1:.4f}  |{bar:<18}|")

    if rea:
        print()
        print("  ── REASONING QUALITY ────────────────────────────────────────────────")
        for metric in ("rouge_l", "bertscore_f1", "step_completeness"):
            if metric in rea and rea[metric] is not None:
                val = rea[metric]
                thr = BENCHMARK_THRESHOLDS.get(metric, 0)
                ok  = "[PASS]" if val >= thr else "[FAIL]"
                print(f"  {metric:24s} : {val}   (threshold {thr})   {ok}")
            elif metric in rea and rea[metric] is None:
                print(f"  {metric:24s} : (not available — see logs)")

    print()
    print("  ── INFERENCE LATENCY ────────────────────────────────────────────────")
    print(f"  latency_p50_ms           : {det['latency_p50_ms']:.2f} ms")
    print(f"  latency_p95_ms           : {det['latency_p95_ms']:.2f} ms")
    print("=" * 72 + "\n")


# ── Public API ───────────────────────────────────────────────────────────────
def run_benchmark(
    *,
    model_path: str = "auto",
    split: str = "test",
    hardware: str = "cpu",
    batch_size: int = 32,
    output_dir: str = ".",
    skip_reasoning: bool = False,
    prefer_offline: bool = False,
    synthetic_n_per_class: int = 30,
    progress: bool = True,
) -> Dict:
    """
    End-to-end benchmark — guaranteed to run to completion.
    """
    t0 = time.time()

    # 1. Load model (always succeeds — falls back to reference)
    model = build_default_model(model_path, hardware=hardware)

    # 2. Load dataset (always succeeds — falls back through 3 tiers)
    dataset = load_dataset(
        split=split,
        prefer_offline=prefer_offline,
        synthetic_n_per_class=synthetic_n_per_class,
    )

    # 3. Detection evaluation
    det = evaluate_detection(model, dataset, batch_size=batch_size, progress=progress)

    # 4. Reasoning (optional)
    rea = None if skip_reasoning else evaluate_reasoning(model, dataset, progress=progress)

    # 5. Compile results
    results = {
        "benchmark":      "QAD-Bench v1.1",
        "dataset":        "TeleAntiFraud-28k",
        "dataset_source": dataset.source.value,
        "model":          model_path,
        "mode":           model.mode,
        "hardware":       hardware,
        "split":          split,
        "n_samples":      len(dataset),
        "timestamp":      datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "wall_time_sec":  round(time.time() - t0, 2),
        "detection":      det,
        "reasoning":      rea,
        "thresholds":     BENCHMARK_THRESHOLDS,
        "citations": {
            "dataset": "Ma et al., ACM MM 2025, pp. 5853-5862",
            "safeqaq": "Wang et al., arXiv:2601.01392, 2026",
            "qad_mg":  "Zhang et al., 2026 (this work)",
        },
    }

    # 6. Save and report
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"qad_bench_{hardware}_{split}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info("Results saved → %s", out_path)

    print_results(results)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="qad_bench",
        description="QAD-Bench v1.1 — Reproducible Telecom Fraud Detection Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Offline reference run (always works)
    python -m qad_bench.runner --model_path auto --prefer_offline --skip_reasoning

    # Real GGUF model on CPU
    python -m qad_bench.runner --model_path ./qad_q4km.gguf --hardware cpu

    # Real PyTorch model on GPU
    python -m qad_bench.runner --model_path ./qad_student.pt --hardware gpu
""",
    )
    parser.add_argument("--model_path",  default="auto",
                        help="Path to .gguf/.pt, or 'auto' for reference model.")
    parser.add_argument("--split",       default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--hardware",    default="cpu",
                        choices=["cpu", "gpu", "mobile"])
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--output_dir",  default=".")
    parser.add_argument("--skip_reasoning", action="store_true")
    parser.add_argument("--prefer_offline", action="store_true",
                        help="Skip HuggingFace lookup and go straight to cache/synthetic.")
    parser.add_argument("--synthetic_n_per_class", type=int, default=30,
                        help="Per-class size for synthetic fallback dataset.")
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--log_level",   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print(f"\n{'='*72}\n  QAD-Bench v1.1 — Telecom Fraud Detection Benchmark\n  Dataset: TeleAntiFraud-28k (Ma et al., ACM MM 2025)\n{'='*72}\n")

    try:
        results = run_benchmark(
            model_path           = args.model_path,
            split                = args.split,
            hardware             = args.hardware,
            batch_size           = args.batch_size,
            output_dir           = args.output_dir,
            skip_reasoning       = args.skip_reasoning,
            prefer_offline       = args.prefer_offline,
            synthetic_n_per_class= args.synthetic_n_per_class,
            progress             = not args.no_progress,
        )
        # Exit code: 0 if all PASS, 1 if any threshold fails
        det = results["detection"]
        passed = (
            det["macro_f1"]    >= BENCHMARK_THRESHOLDS["macro_f1"] and
            det["weighted_f1"] >= BENCHMARK_THRESHOLDS["weighted_f1"]
        )
        return 0 if passed else 1
    except Exception as exc:                       # noqa: BLE001
        log.error("FATAL: %s", exc, exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
