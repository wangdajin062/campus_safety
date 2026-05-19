"""
tests/test_v41_features.py — QAD-MultiGuard v4.1 新功能单元测试
=================================================================
覆盖 6 个差距修复 + 安全修复的单元级验证。

运行: pytest tests/test_v41_features.py -v
"""
import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════
# 差距修复 #4: AcousticFeatures.voice_risk_score()
# ═══════════════════════════════════════════════════════════════
class TestVoiceRiskScore:
    """论文 §IV.A 声学风险评分 [0, 100]"""

    def test_zero_inputs_gives_zero(self):
        from ml.acoustic_embedding import AcousticFeatures
        feat = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.0,
            energy_var=0.0, tone_proxy=0.0, urgency_proxy=0.0, pitch_range=0.0,
        )
        assert feat.voice_risk_score() == 0

    def test_max_inputs_gives_100(self):
        from ml.acoustic_embedding import AcousticFeatures
        # 最大输入 → 各分量上限
        feat = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.0,
            energy_var=1.0, tone_proxy=1.0, urgency_proxy=1.0, pitch_range=1.0,
        )
        # 35 + 28 + 25 + 12 = 100, 但 min(100, ...) 保证上限
        score = feat.voice_risk_score()
        assert 0 <= score <= 100, f"voice_risk_score should be [0,100], got {score}"
        assert score == 100, f"expected 100, got {score}"

    def test_output_range_is_always_0_to_100(self):
        from ml.acoustic_embedding import AcousticFeatures
        for ev in [0.0, 0.5, 2.0, -0.1]:
            for tp in [0.0, 0.5, 2.0]:
                feat = AcousticFeatures(
                    embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
                    duration_s=1.0,
                    energy_var=ev, tone_proxy=tp, urgency_proxy=0.0, pitch_range=0.0,
                )
                s = feat.voice_risk_score()
                assert 0 <= s <= 100, f"score {s} out of [0,100]"

    def test_high_energy_var_increases_score(self):
        from ml.acoustic_embedding import AcousticFeatures
        low = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.0, energy_var=0.1, tone_proxy=0, urgency_proxy=0, pitch_range=0,
        )
        high = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.0, energy_var=0.9, tone_proxy=0, urgency_proxy=0, pitch_range=0,
        )
        assert high.voice_risk_score() > low.voice_risk_score()


# ═══════════════════════════════════════════════════════════════
# 差距修复 #4: acoustic_indicators()
# ═══════════════════════════════════════════════════════════════
class TestAcousticIndicators:
    """韵律指标字典验证"""

    def test_indicators_contains_all_keys(self):
        from ml.acoustic_embedding import AcousticFeatures
        feat = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.5,
            energy_var=0.3, tone_proxy=0.4, urgency_proxy=0.2, pitch_range=0.1,
            is_dp=True, dp_epsilon=9.69,
        )
        ind = feat.acoustic_indicators()
        required_keys = {"energy_variance", "tone_proxy", "urgency_proxy",
                         "pitch_range", "voice_risk_score", "duration_s", "dp_epsilon"}
        assert required_keys.issubset(ind.keys()), f"Missing keys: {required_keys - ind.keys()}"

    def test_dp_epsilon_none_when_dp_disabled(self):
        from ml.acoustic_embedding import AcousticFeatures
        feat = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.0, is_dp=False,
            energy_var=0, tone_proxy=0, urgency_proxy=0, pitch_range=0,
        )
        assert feat.acoustic_indicators()["dp_epsilon"] is None


# ═══════════════════════════════════════════════════════════════
# 差距修复 #2: URL 特征评分
# ═══════════════════════════════════════════════════════════════
class TestUrlScoring:
    """多模态检测器 URL 评分逻辑（w_url=0.20）"""

    @pytest.mark.asyncio
    async def test_ip_address_url_scores_high(self):
        from ml.multimodal_detector import MultimodalDetector, MultimodalInput
        detector = MultimodalDetector()
        inp = MultimodalInput(
            url_features=[0.5, 0.3, 1.0, 0.0, 0.5, 0.0],  # has_ip=1.0
        )
        result = await detector._fast_detect(inp)
        assert result.url_score >= 40, f"IP URL should score >= 40, got {result.url_score}"

    @pytest.mark.asyncio
    async def test_normal_url_scores_low(self):
        from ml.multimodal_detector import MultimodalDetector, MultimodalInput
        detector = MultimodalDetector()
        inp = MultimodalInput(
            url_features=[0.3, 0.1, 0.0, 0.0, 0.2, 0.0],  # no red flags
        )
        result = await detector._fast_detect(inp)
        assert result.url_score < 30, f"Normal URL should score < 30, got {result.url_score}"

    @pytest.mark.asyncio
    async def test_url_score_never_exceeds_100(self):
        from ml.multimodal_detector import MultimodalDetector, MultimodalInput
        detector = MultimodalDetector()
        inp = MultimodalInput(
            url_features=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # all max
        )
        result = await detector._fast_detect(inp)
        assert 0 <= result.url_score <= 100

    @pytest.mark.asyncio
    async def test_url_score_integrated_into_fusion(self):
        """验证 url_score 参与 L-BFGS 融合（之前总是 0）"""
        from ml.multimodal_detector import MultimodalDetector, MultimodalInput
        detector = MultimodalDetector()
        inp_with_url = MultimodalInput(
            sms_features=[1.0]*12,
            url_features=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # max URL risk
        )
        result = await detector._fast_detect(inp_with_url)
        assert result.fused_score_lbfgs > 0, "URL features should contribute to fusion"

    @pytest.mark.asyncio
    async def test_url_keyword_fallback_no_features(self):
        """无结构化特征时，通过 URL 字符串做关键词检查"""
        from ml.multimodal_detector import MultimodalDetector, MultimodalInput
        detector = MultimodalDetector()
        inp = MultimodalInput(urls=["https://bit.ly/xyz123"])
        result = await detector._fast_detect(inp)
        assert result.url_score >= 45, "Short URL should score >= 45 in fallback"


# ═══════════════════════════════════════════════════════════════
# 差距修复 #1 & #3: extract_from_embedding_list
# ═══════════════════════════════════════════════════════════════
class TestExtractFromEmbeddingList:
    """从 Android 上报嵌入重建特征容器"""

    def test_reconstructs_features_from_128d(self):
        from ml.acoustic_embedding import (
            AcousticEmbeddingExtractor, EMBEDDING_DIM, MFCC_DIM,
        )
        rng = np.random.default_rng(42)
        embedding = rng.normal(0, 0.1, EMBEDDING_DIM).tolist()
        feat = AcousticEmbeddingExtractor().extract_from_embedding_list(embedding)
        assert feat.f_mfcc.shape == (MFCC_DIM,)
        assert feat.f_proj.shape == (EMBEDDING_DIM - MFCC_DIM,)
        assert feat.embedding.shape == (EMBEDDING_DIM,)
        assert feat.voice_risk_score() is not None

    def test_pads_short_embedding(self):
        from ml.acoustic_embedding import (
            AcousticEmbeddingExtractor, EMBEDDING_DIM,
        )
        short = [0.0] * 8  # too short
        feat = AcousticEmbeddingExtractor().extract_from_embedding_list(short)
        assert len(feat.embedding) == EMBEDDING_DIM

    def test_voice_risk_score_works_after_reconstruction(self):
        from ml.acoustic_embedding import AcousticEmbeddingExtractor
        rng = np.random.default_rng(42)
        emb = rng.normal(0, 0.1, 128).tolist()
        feat = AcousticEmbeddingExtractor().extract_from_embedding_list(emb)
        score = feat.voice_risk_score()
        assert isinstance(score, int)
        assert 0 <= score <= 100


# ═══════════════════════════════════════════════════════════════
# 差距修复 #3: WhisperProjection 行归一化
# ═══════════════════════════════════════════════════════════════
class TestWhisperProjectionRowNorm:
    """验证 W_proj 每行 L2 归一化"""

    def test_rows_are_unit_norm(self):
        from ml.acoustic_embedding import WhisperProjection
        proj = WhisperProjection()
        row_norms = np.linalg.norm(proj.W, axis=1)
        assert np.allclose(row_norms, 1.0, atol=1e-6), \
            f"Row norms not ~1.0: {row_norms[:5]}"

    def test_projection_output_is_l2_normalized(self):
        from ml.acoustic_embedding import WhisperProjection
        proj = WhisperProjection()
        rng = np.random.default_rng(42)
        cls_vec = rng.normal(0, 1, 384)
        out = proj.project(cls_vec)
        norm = np.linalg.norm(out)
        assert abs(norm - 1.0) < 1e-6, f"Output norm {norm} != 1.0"


# ═══════════════════════════════════════════════════════════════
# 差距修复 #3: MFCCExtractor 韵律分解（4 路）
# ═══════════════════════════════════════════════════════════════
class TestProsodyDecomposition:
    """4 路粗粒度韵律分解验证"""

    def test_prosody_returns_expected_keys(self):
        from ml.acoustic_embedding import MFCCExtractor
        ext = MFCCExtractor()
        rng = np.random.default_rng(42)
        pcm = rng.normal(0, 0.1, 16000).astype(np.float32)  # 1s audio
        _, prosody = ext.extract(pcm)
        assert "energy_var" in prosody
        assert "tone_proxy" in prosody
        assert "urgency_proxy" in prosody
        assert "pitch_range" in prosody

    def test_short_audio_returns_zero_prosody(self):
        from ml.acoustic_embedding import MFCCExtractor
        ext = MFCCExtractor()
        pcm = np.zeros(100, dtype=np.float32)  # too short
        _, prosody = ext.extract(pcm)
        assert prosody["energy_var"] == 0.0

    def test_variable_audio_generates_nonzero_prosody(self):
        from ml.acoustic_embedding import MFCCExtractor
        ext = MFCCExtractor()
        # PCM with varying amplitude
        t = np.linspace(0, 1, 16000, endpoint=False)
        pcm = (np.sin(2 * np.pi * 440 * t) * 0.5 +
               np.sin(2 * np.pi * 880 * t) * 0.3).astype(np.float32)
        _, prosody = ext.extract(pcm)
        assert prosody["energy_var"] > 0, "Energy variance should be >0 for tonal audio"
        assert prosody["tone_proxy"] > 0, "Tone proxy should be >0"


# ═══════════════════════════════════════════════════════════════
# H6: DP Epsilon 计算验证
# ═══════════════════════════════════════════════════════════════
class TestDpEpsilon:
    """论文 §III.B DP 高斯机制正确实现"""

    def test_dp_epsilon_formula(self):
        from ml.acoustic_embedding import calc_dp_epsilon
        # σ=1.0, Δ₂=2.0, δ=1e-5
        # ε = Δ₂ · sqrt(2 · ln(1.25/δ)) / σ
        # Expected: 2.0 * sqrt(2 * ln(1.25/1e-5)) / 1.0 ≈ 9.69
        eps = calc_dp_epsilon(1.0)
        assert abs(eps - 9.69) < 0.1, f"Expected ~9.69, got {eps}"

    def test_sigma_zero_returns_inf(self):
        from ml.acoustic_embedding import calc_dp_epsilon
        assert calc_dp_epsilon(0.0) == float("inf")
        assert calc_dp_epsilon(-1.0) == float("inf")

    def test_smaller_sigma_gives_larger_epsilon(self):
        from ml.acoustic_embedding import calc_dp_epsilon
        eps_large_sigma = calc_dp_epsilon(2.0)
        eps_small_sigma = calc_dp_epsilon(0.5)
        assert eps_small_sigma > eps_large_sigma


# ═══════════════════════════════════════════════════════════════
# H2: Retrain 限流标志位检查
# ═══════════════════════════════════════════════════════════════
class TestRetrainDedup:
    """验证 retrain 去重标志"""

    def test_retrain_in_progress_exists(self):
        import importlib
        inf = importlib.import_module("api.v1.inference")
        importlib.reload(inf)
        assert hasattr(inf, "_retrain_in_progress")
        assert inf._retrain_in_progress is False  # initial state


# ═══════════════════════════════════════════════════════════════
# C1: 死代码验证（模块级无 body 引用）
# ═══════════════════════════════════════════════════════════════
class TestNoDeadCode:
    """验证 inference.py 无死代码"""

    def test_no_duplicate_routes(self):
        import os
        inf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "api", "v1", "inference.py")
        with open(inf_path, encoding="utf-8") as f:
            content = f.read()
        assert content.count("async def infer_fast(") == 1
        assert content.count("async def analyze_voice(") == 1
        assert "body.sms_features" not in content.split("def ")[0]

    def test_file_shorter_than_767(self):
        """修复后文件从 767 行缩减"""
        import os
        inf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                "api", "v1", "inference.py")
        with open(inf_path, encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) < 600, f"File too long: {len(lines)} lines"
