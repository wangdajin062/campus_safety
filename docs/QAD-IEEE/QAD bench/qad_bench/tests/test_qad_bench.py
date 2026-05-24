"""
Test suite for QAD-Bench v1.1.

Engineered as test-engineer would: every failure path the original
script suffered is now a regression test.
"""
import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

# Make `qad_bench` importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qad_bench import (
    QADMultiGuardModel, build_default_model,
    load_dataset, DatasetSource,
    extract_features, compute_mfcc, compute_whisper_embedding,
    evaluate_detection, evaluate_reasoning,
    run_benchmark,
    FRAUD_CATEGORIES, BENCHMARK_THRESHOLDS,
)
from qad_bench.constants import FEATURE_DIM


# ═════════════════════════════════════════════════════════════════════════════
# 1. Dataset loader regression tests
# ═════════════════════════════════════════════════════════════════════════════
class TestDatasetLoader(unittest.TestCase):
    """Regression: original script crashed when HF was unreachable."""

    def test_offline_synthetic_fallback(self):
        """When prefer_offline=True, returns synthetic dataset."""
        ds = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=10)
        self.assertEqual(ds.source, DatasetSource.SYNTHETIC)
        self.assertGreater(len(ds), 0)

    def test_synthetic_label_distribution(self):
        """Synthetic data must include all 10 classes (9 fraud + non-fraud)."""
        ds = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=20)
        labels = sorted({s.category_id for s in ds})
        self.assertEqual(labels, list(range(10)))

    def test_synthetic_sample_schema(self):
        """Each sample must have all required fields."""
        ds = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=5)
        sample = ds[0]
        for field in ("sample_id", "audio", "transcript", "category_id",
                      "is_fraud", "cot_annotation"):
            self.assertTrue(hasattr(sample, field), f"Missing field {field}")

    def test_synthetic_audio_shape(self):
        ds = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=3)
        for s in ds:
            self.assertIn("array", s.audio)
            self.assertEqual(s.audio["sampling_rate"], 16000)
            self.assertGreater(len(s.audio["array"]), 0)

    def test_split_validation(self):
        with self.assertRaises(ValueError):
            load_dataset(split="invalid_split")

    def test_select_returns_subset(self):
        ds = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=10)
        sub = ds.select(range(5))
        self.assertEqual(len(sub), 5)

    def test_synthetic_reproducibility(self):
        """Same seed → identical samples."""
        ds1 = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=5, seed=99)
        ds2 = load_dataset(split="test", prefer_offline=True, synthetic_n_per_class=5, seed=99)
        ids1 = [s.sample_id for s in ds1]
        ids2 = [s.sample_id for s in ds2]
        self.assertEqual(ids1, ids2)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Feature extraction tests
# ═════════════════════════════════════════════════════════════════════════════
class TestFeatures(unittest.TestCase):
    """Regression: features must extract even when librosa/transformers missing."""

    def test_mfcc_dimension(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000).astype(np.float32)
        mfcc = compute_mfcc(audio)
        self.assertEqual(mfcc.shape, (64,))
        self.assertEqual(mfcc.dtype, np.float32)

    def test_mfcc_empty_audio(self):
        mfcc = compute_mfcc(np.zeros(0, dtype=np.float32))
        self.assertEqual(mfcc.shape, (64,))

    def test_whisper_embedding_dimension(self):
        rng = np.random.default_rng(0)
        audio = rng.standard_normal(16000).astype(np.float32)
        emb = compute_whisper_embedding(audio, transcript="测试")
        self.assertEqual(emb.shape, (64,))

    def test_whisper_fallback_with_transcript(self):
        """Hash fallback must produce different embeddings for different text."""
        e1 = compute_whisper_embedding(np.zeros(0), transcript="您好，公安局")
        e2 = compute_whisper_embedding(np.zeros(0), transcript="妈我下周回家")
        self.assertFalse(np.allclose(e1, e2))

    def test_extract_features_full_dim(self):
        sample = {"audio": {"array": np.zeros(16000, dtype=np.float32),
                            "sampling_rate": 16000},
                  "transcript": "您好这里是公安局，您涉嫌洗钱"}
        feat = extract_features(sample)
        self.assertEqual(feat.shape, (FEATURE_DIM,))
        self.assertEqual(feat.dtype, np.float32)

    def test_extract_features_handles_missing_fields(self):
        """Should not raise even when audio dict is empty."""
        feat = extract_features({"transcript": "spam"})
        self.assertEqual(feat.shape, (FEATURE_DIM,))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Model loading regression tests
# ═════════════════════════════════════════════════════════════════════════════
class TestModelLoader(unittest.TestCase):
    """Regression: original crashed on missing model."""

    def test_auto_model_always_works(self):
        """`auto` mode never needs a file."""
        m = build_default_model("auto")
        self.assertEqual(m.mode, "reference")

    def test_missing_file_falls_back_gracefully(self):
        """Original behavior: FileNotFoundError. New: warns and uses reference."""
        m = build_default_model("/nonexistent/path/model.gguf")
        self.assertEqual(m.mode, "reference")  # NOT a crash

    def test_unknown_extension_falls_back(self):
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            m = build_default_model(f.name)
        self.assertEqual(m.mode, "reference")

    def test_predict_proba_returns_distribution(self):
        m = build_default_model("auto")
        feats = np.zeros((4, FEATURE_DIM), dtype=np.float32)
        probs = m.predict_proba(feats, transcripts=["", "", "", ""])
        self.assertEqual(probs.shape, (4, 10))
        # Each row sums to ~1
        for row in probs:
            self.assertAlmostEqual(row.sum(), 1.0, places=5)

    def test_keyword_routing(self):
        """Public-security keywords should bias toward category 0."""
        m = build_default_model("auto")
        feats = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        probs = m.predict_proba(feats, transcripts=[
            "您好我是公安局民警，您涉嫌洗钱，请将资金转入安全账户配合调查"
        ])
        pred = int(np.argmax(probs[0]))
        self.assertEqual(pred, 0, f"Expected public_security, got {FRAUD_CATEGORIES[pred] if pred < 9 else 'non_fraud'}")

    def test_part_time_job_routing(self):
        m = build_default_model("auto")
        feats = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        probs = m.predict_proba(feats, transcripts=[
            "兼职刷单日结，垫付本金返佣10%"
        ])
        pred = int(np.argmax(probs[0]))
        self.assertEqual(pred, 2)

    def test_non_fraud_routing(self):
        m = build_default_model("auto")
        feats = np.zeros((1, FEATURE_DIM), dtype=np.float32)
        probs = m.predict_proba(feats, transcripts=["妈我下周回家吃饭"])
        pred = int(np.argmax(probs[0]))
        self.assertEqual(pred, 9)  # non_fraud

    def test_cot_generation_3_steps(self):
        m = build_default_model("auto")
        cot = m.generate_cot(transcript="您涉嫌洗钱请转入安全账户")
        for marker in ("步骤1", "步骤2", "步骤3"):
            self.assertIn(marker, cot)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Evaluator tests
# ═════════════════════════════════════════════════════════════════════════════
class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.dataset = load_dataset(split="test", prefer_offline=True,
                                    synthetic_n_per_class=15, seed=7)
        self.model   = build_default_model("auto")

    def test_detection_returns_all_required_fields(self):
        result = evaluate_detection(self.model, self.dataset, batch_size=8, progress=False)
        for field in ("macro_f1", "weighted_f1", "auc_roc",
                      "per_category_f1", "latency_p50_ms", "latency_p95_ms",
                      "n_samples"):
            self.assertIn(field, result)

    def test_detection_macro_f1_in_range(self):
        result = evaluate_detection(self.model, self.dataset, batch_size=8, progress=False)
        self.assertGreaterEqual(result["macro_f1"], 0.0)
        self.assertLessEqual(result["macro_f1"], 1.0)

    def test_detection_realistic_offline_f1(self):
        """Offline reference + synthetic data should hit threshold (≥0.80)."""
        result = evaluate_detection(self.model, self.dataset, batch_size=8, progress=False)
        self.assertGreaterEqual(
            result["macro_f1"], BENCHMARK_THRESHOLDS["macro_f1"],
            f"macro_f1={result['macro_f1']} below threshold")

    def test_detection_per_category_complete(self):
        result = evaluate_detection(self.model, self.dataset, batch_size=8, progress=False)
        self.assertEqual(set(result["per_category_f1"].keys()), set(FRAUD_CATEGORIES))

    def test_reasoning_returns_required_fields(self):
        result = evaluate_reasoning(self.model, self.dataset, max_samples=20, progress=False)
        for field in ("rouge_l", "step_completeness", "n_reasoning_samples"):
            self.assertIn(field, result)

    def test_reasoning_step_completeness_high(self):
        """Reference CoT always has all 3 steps → completeness should be ≥ 90%."""
        result = evaluate_reasoning(self.model, self.dataset, max_samples=30, progress=False)
        self.assertGreaterEqual(result["step_completeness"], 90.0)


# ═════════════════════════════════════════════════════════════════════════════
# 5. End-to-end runner tests (the original failure mode!)
# ═════════════════════════════════════════════════════════════════════════════
class TestRunnerE2E(unittest.TestCase):
    """The CRITICAL regression: `qad_bench_eval.py --model_path X.gguf` crashed.
    The new runner must always complete."""

    def test_run_benchmark_offline_no_model(self):
        """E2E: no model file, no internet — must still produce results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_benchmark(
                model_path     = "auto",
                split          = "test",
                hardware       = "cpu",
                batch_size     = 16,
                output_dir     = tmpdir,
                skip_reasoning = True,
                prefer_offline = True,
                synthetic_n_per_class = 12,
                progress       = False,
            )
            self.assertIn("detection", results)
            self.assertGreaterEqual(results["detection"]["macro_f1"], 0.80)
            # JSON output must exist
            jsons = list(Path(tmpdir).glob("*.json"))
            self.assertEqual(len(jsons), 1)
            with open(jsons[0]) as f:
                json.load(f)   # must be valid JSON

    def test_run_benchmark_with_missing_model_path(self):
        """E2E with explicitly missing model — original would die here."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_benchmark(
                model_path     = "/this/file/does/not/exist.gguf",
                split          = "test",
                hardware       = "cpu",
                batch_size     = 8,
                output_dir     = tmpdir,
                skip_reasoning = True,
                prefer_offline = True,
                synthetic_n_per_class = 8,
                progress       = False,
            )
            self.assertEqual(results["mode"], "reference")
            self.assertIn("detection", results)

    def test_run_benchmark_full_with_reasoning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = run_benchmark(
                model_path     = "auto",
                hardware       = "cpu",
                output_dir     = tmpdir,
                prefer_offline = True,
                synthetic_n_per_class = 10,
                progress       = False,
            )
            self.assertIsNotNone(results["reasoning"])
            self.assertIn("rouge_l", results["reasoning"])
            self.assertIn("step_completeness", results["reasoning"])


# ═════════════════════════════════════════════════════════════════════════════
# 6. CLI smoke test
# ═════════════════════════════════════════════════════════════════════════════
class TestCLI(unittest.TestCase):
    """Make sure the python -m entrypoint works."""

    def test_cli_help(self):
        from qad_bench import runner
        with self.assertRaises(SystemExit) as ctx:
            runner.main(["--help"])
        self.assertEqual(ctx.exception.code, 0)

    def test_cli_offline_run(self):
        from qad_bench import runner
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = runner.main([
                "--model_path", "auto",
                "--prefer_offline",
                "--skip_reasoning",
                "--no_progress",
                "--synthetic_n_per_class", "10",
                "--output_dir", tmpdir,
                "--log_level", "WARNING",
            ])
            self.assertEqual(rc, 0, f"CLI exited with {rc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
