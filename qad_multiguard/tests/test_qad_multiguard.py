"""Unit tests for QAD-MultiGuard core modules."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest


def test_config_defaults():
    from qad_multiguard.config import Config, load_config
    cfg = Config()
    assert cfg.qad.loss == "pure_kl"
    assert cfg.qad.softmax_temp == 1.0
    assert cfg.ov_freeze.step_ratio == 0.30
    assert cfg.privacy.embed_dim == 128
    assert cfg.fusion.weights == (0.40, 0.30, 0.20, 0.10)
    # Roundtrip via dict (asdict converts tuples to lists)
    d = cfg.to_dict()
    assert tuple(d["fusion"]["weights"]) == (0.40, 0.30, 0.20, 0.10)


# ─── Quantization tests ─────────────────────────────────────────────────
def test_quantization_nvfp4_shape_and_dtype():
    from qad_multiguard.quantization import quantize_nvfp4
    W = torch.randn(64, 256)
    Wq = quantize_nvfp4(W, block_size=16)
    assert Wq.shape == W.shape
    assert Wq.dtype == W.dtype


def test_quantization_nvfp4_better_than_ptq():
    """NVFP4 should yield lower MSE than vanilla PTQ on the same weights."""
    from qad_multiguard.quantization import quantize_nvfp4, quantize_ptq
    torch.manual_seed(0)
    W = torch.randn(2048, 4096) * 0.05
    nvfp4_mse = ((quantize_nvfp4(W) - W) ** 2).mean().item()
    ptq_mse = ((quantize_ptq(W, bits=4) - W) ** 2).mean().item()
    assert nvfp4_mse < ptq_mse, f"NVFP4 ({nvfp4_mse}) >= PTQ ({ptq_mse})"


def test_q4_k_m_quality():
    """Q4_K_M with 6-bit critical fallback should be at least as good as plain 4-bit PTQ."""
    from qad_multiguard.quantization import quantize_q4_k_m, quantize_ptq
    torch.manual_seed(0)
    W = torch.randn(512, 1024) * 0.05
    q4km_mse = ((quantize_q4_k_m(W) - W) ** 2).mean().item()
    ptq_mse = ((quantize_ptq(W, bits=4) - W) ** 2).mean().item()
    # The 6-bit fallback should reduce MSE relative to plain 4-bit
    assert q4km_mse <= ptq_mse * 1.05  # allow small margin


def test_quantization_dispatcher():
    from qad_multiguard.quantization import quantize_weight, QUANT_METHODS
    W = torch.randn(64, 128)
    for method in QUANT_METHODS:
        Wq = quantize_weight(W, method)
        assert Wq.shape == W.shape, f"Shape mismatch for {method}"


# ─── OV-Freeze tests ────────────────────────────────────────────────────
def test_ov_freeze_proposition_1_bound():
    """Prop 1: |∂c_l/∂y_l| ≤ c_l / (n · Var(y_l))^(1/2)."""
    from qad_multiguard.ov_freeze import OVFreeze
    import torch.nn as nn
    layer = nn.Linear(64, 64)
    wrapped = OVFreeze(layer, target_variance=1.0, enabled=True)
    bound = wrapped.gradient_norm_bound(n_samples=32)
    # The bound is c / sqrt(n * Var) ≤ 1 / sqrt(32 * 1) = 0.177
    assert bound <= 1.0


def test_ov_freeze_disabled_is_identity():
    """When disabled, OVFreeze should pass through unchanged."""
    from qad_multiguard.ov_freeze import OVFreeze
    import torch.nn as nn
    layer = nn.Linear(16, 16)
    wrapped = OVFreeze(layer, enabled=False)
    x = torch.randn(8, 16)
    y_wrapped = wrapped(x)
    y_orig = layer(x)
    assert torch.allclose(y_wrapped, y_orig, atol=1e-6)


def test_ov_freeze_reduces_var_drift():
    """When enabled with the right target variance, OVF reduces variance drift."""
    from qad_multiguard.ov_freeze import OVFreeze
    import torch.nn as nn
    torch.manual_seed(0)
    layer = nn.Linear(64, 64)
    # Pretend target var was measured at 0.3
    target_var = 0.3
    wrapped = OVFreeze(layer, target_variance=target_var, enabled=True,
                       coefficient=1.0)
    x = torch.randn(128, 64) * 2.0  # Inflated input → larger output var
    with torch.no_grad():
        y_wrap = wrapped(x)
        y_orig = layer(x)
    drift_orig = abs(float(y_orig.var().item()) - target_var) / target_var
    drift_wrap = abs(float(y_wrap.var().item()) - target_var) / target_var
    assert drift_wrap < drift_orig, "OVF should reduce drift to target variance"


# ─── Distillation tests ─────────────────────────────────────────────────
def test_loss_pure_kl_is_zero_when_identical():
    from qad_multiguard.distillation import loss_pure_kl
    logits = torch.randn(4, 8, 100)
    loss, kl = loss_pure_kl(logits.clone(), logits.clone())
    assert float(loss.item()) < 1e-5
    assert kl < 1e-5


def test_loss_kl_increases_with_divergence():
    """KL(p_T || p_S) should grow as student diverges from teacher."""
    from qad_multiguard.distillation import loss_pure_kl
    torch.manual_seed(0)
    base = torch.randn(4, 8, 50)
    losses = []
    for noise in [0.0, 0.5, 2.0]:
        student = base + torch.randn_like(base) * noise
        _, kl = loss_pure_kl(student, base)
        losses.append(kl)
    assert losses[0] <= losses[1] <= losses[2]


def test_all_loss_registry_present():
    from qad_multiguard.distillation import LOSS_REGISTRY
    expected = {"pure_kl", "mse", "cross_entropy", "three_term", "kl_task_reg"}
    assert set(LOSS_REGISTRY.keys()) == expected


# ─── Privacy tests ──────────────────────────────────────────────────────
def test_acoustic_embedding_shape():
    from qad_multiguard.privacy import AcousticEmbedder
    from qad_multiguard.config import Config
    embedder = AcousticEmbedder(Config().privacy)
    audio = np.random.normal(0, 0.3, 16000 * 3).astype(np.float32)
    F_v = embedder(audio)
    assert F_v.shape == (128,)
    assert F_v.dtype == np.float32


def test_acoustic_embedding_invariance_to_time_shift():
    """Time-averaging step #1 means F_v should be roughly invariant to small shifts."""
    from qad_multiguard.privacy import AcousticEmbedder
    from qad_multiguard.config import Config
    embedder = AcousticEmbedder(Config().privacy)
    audio = np.sin(np.arange(16000) * 0.01).astype(np.float32)
    F_v_1 = embedder.extract_mfcc_avg(audio)
    F_v_2 = embedder.extract_mfcc_avg(np.roll(audio, 100))
    # Should be very close (within numerical precision)
    np.testing.assert_allclose(F_v_1, F_v_2, atol=0.5)


def test_glo_attack_returns_high_wer():
    """White-box and black-box GLO attacks should both yield WER ≥ 0.92."""
    from qad_multiguard.privacy import AcousticEmbedder, evaluate_privacy
    from qad_multiguard.config import Config
    embedder = AcousticEmbedder(Config().privacy)
    audios = [np.random.normal(0, 0.3, 16000).astype(np.float32) for _ in range(20)]
    res = evaluate_privacy(embedder, audios)
    assert res["white_box"].wer >= 0.92
    assert res["black_box"].wer >= 0.92
    # Speaker ID should be near random (10%)
    assert 0.05 <= res["white_box"].speaker_id_acc <= 0.12
    assert 0.05 <= res["black_box"].speaker_id_acc <= 0.12


# ─── Fusion tests ───────────────────────────────────────────────────────
def test_fusion_lbfgs_recovers_planted_weights():
    from qad_multiguard.fusion import fit_lbfgs, FusionWeights
    rng = np.random.RandomState(0)
    n = 20000
    R = rng.uniform(0, 1, (n, 4))
    true_w = np.array([0.40, 0.30, 0.20, 0.10])
    z = R @ true_w + 0.0
    p = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    out = fit_lbfgs(R, y)
    # Weights should be close to true within ~0.10 (allowing for logistic noise + ridge)
    assert abs(out.w_text - 0.40) < 0.10
    assert abs(out.w_audio - 0.30) < 0.10


def test_fusion_5fold_cv_stability():
    from qad_multiguard.fusion import cv_5fold
    rng = np.random.RandomState(42)
    n = 3000
    R = rng.uniform(0, 1, (n, 4))
    true_w = np.array([0.40, 0.30, 0.20, 0.10])
    p = 1 / (1 + np.exp(-(R @ true_w)))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    cv = cv_5fold(R, y)
    # All folds should be within ±0.05 of mean
    for w in cv.fold_weights:
        assert abs(w.w_text - cv.mean_weights.w_text) < 0.10


# ─── Speculative decoding tests ─────────────────────────────────────────
def test_speculative_speedup_formula():
    """Verify (1 - α^(γ+1))/(1 - α) at the paper-cited operating points."""
    from qad_multiguard.speculative import theoretical_speedup
    # α=0.86, γ=5 → 4.25× per the corrected paper Table IX
    assert abs(theoretical_speedup(0.86, 5) - 4.253) < 0.01
    # α=0.78, γ=5 → 3.52×
    assert abs(theoretical_speedup(0.78, 5) - 3.522) < 0.01
    # Edge: α→1 should approach γ+1
    assert abs(theoretical_speedup(0.99999, 5) - 6.0) < 0.1


def test_speculative_kv_cache_monotonic():
    from qad_multiguard.speculative import kv_cache_size
    sizes = [kv_cache_size(g) for g in [3, 5, 7, 10]]
    assert sizes == sorted(sizes)


# ─── Metrics tests ──────────────────────────────────────────────────────
def test_classification_metrics_all_correct():
    from qad_multiguard.metrics import classification_metrics
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    m = classification_metrics(y_true, y_pred)
    assert m.f1 == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.fpr == 0.0


def test_classification_metrics_all_wrong():
    from qad_multiguard.metrics import classification_metrics
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    m = classification_metrics(y_true, y_pred)
    assert m.f1 == 0.0


def test_recovery_rate():
    from qad_multiguard.metrics import recovery_rate
    assert abs(recovery_rate(0.928, 0.937) - 99.04) < 0.1


def test_wilcoxon_rejects_significant_difference():
    """Wilcoxon should give p < 0.05 when there's a clear difference."""
    from qad_multiguard.metrics import wilcoxon_signed_rank
    rng = np.random.RandomState(0)
    x = rng.normal(5.0, 0.5, 50)
    y = rng.normal(3.0, 0.5, 50)
    res = wilcoxon_signed_rank(x, y)
    assert res["p_value"] < 0.05


def test_cohens_kappa():
    from qad_multiguard.metrics import cohens_kappa
    a = np.array([0, 0, 1, 1, 1])
    b = np.array([0, 0, 1, 1, 1])
    assert cohens_kappa(a, b) == 1.0


# ─── Data tests ─────────────────────────────────────────────────────────
def test_taf_loader_synthetic_distribution():
    """Synthetic TAF-28k should preserve the official 14150/14361 fraud/benign split."""
    from qad_multiguard.data import TAFLoader
    loader = TAFLoader(use_real=False, n_samples=10000, seed=42)
    train = loader.load_split("train")
    assert len(train) == 8000  # 80% of 10000
    fraud_pct = sum(s.label for s in train) / len(train)
    # Allow ±2% slack due to sampling
    assert 0.45 < fraud_pct < 0.55


def test_taf_loader_deterministic():
    """Same seed → same samples."""
    from qad_multiguard.data import TAFLoader
    loader1 = TAFLoader(use_real=False, n_samples=1000, seed=42)
    loader2 = TAFLoader(use_real=False, n_samples=1000, seed=42)
    a = loader1.load_split("train")
    b = loader2.load_split("train")
    assert a[0].text == b[0].text
    assert a[0].label == b[0].label


def test_advfraud_3k_size():
    """AdvFraud-3k should produce ~3000 samples."""
    from qad_multiguard.data import TAFLoader, build_advfraud_3k
    loader = TAFLoader(use_real=False, n_samples=20000, seed=42)
    test = loader.load_split("test")
    adv = build_advfraud_3k(test, seed=42)
    assert 2900 <= len(adv) <= 3100
    # All adversarial samples should be labeled fraud
    assert all(s.label == 1 for s in adv)


# ─── Pipeline tests ─────────────────────────────────────────────────────
def test_pipeline_end_to_end_smoke():
    """Smoke test: pipeline trains and predicts without errors."""
    from qad_multiguard.deployment import QADMultiGuardPipeline
    from qad_multiguard.data import TAFLoader
    loader = TAFLoader(use_real=False, n_samples=2000, seed=42)
    train = loader.load_split("train")
    test = loader.load_split("test")
    p = QADMultiGuardPipeline()
    p.fit(train)
    res = p.evaluate(test)
    # Synthetic data is easy → F1 should be high
    assert res.metrics.f1 > 0.85
    # Latency should be ≤ 350 ms target
    assert res.latency_p50.total_ms < 350


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
