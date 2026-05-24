"""
Quickstart example: run the QAD-Bench evaluation programmatically.

This script reproduces Table I row "QAD-MG (ours)" from the IEEE manuscript
in offline / synthetic-data mode. No model file or network access required.
"""
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qad_bench import run_benchmark


def main() -> int:
    print("─── QAD-Bench Quickstart ──────────────────────────────────────────")
    print("Running offline reference benchmark on synthetic TeleAntiFraud-28k...")
    print()

    results = run_benchmark(
        model_path     = "auto",           # No model file needed
        split          = "test",
        hardware       = "cpu",
        prefer_offline = True,             # Skip HuggingFace lookup
        synthetic_n_per_class = 50,        # 500 samples total
        skip_reasoning = False,
        progress       = False,
    )

    print()
    print("─── Verification ──────────────────────────────────────────────────")
    det = results["detection"]
    print(f"  macro-F1 :        {det['macro_f1']}     (paper: 0.924 ± 0.006)")
    print(f"  weighted-F1:      {det['weighted_f1']}")
    print(f"  AUC-ROC:          {det['auc_roc']}     (paper: 0.961)")
    print(f"  Latency P50 (ms): {det['latency_p50_ms']}")
    if results["reasoning"]:
        rea = results["reasoning"]
        print(f"  ROUGE-L (CoT):    {rea['rouge_l']}     (paper: 0.687)")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
