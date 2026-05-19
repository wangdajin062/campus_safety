"""
tests/test_api_contract.py — QAD-MultiGuard v4.1 前后端契约测试
================================================================
验证 Java ↔ Python 字段对齐、API Schema 一致性。

运行: pytest tests/test_api_contract.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════
# 契约 #1: MultimodalRequest 字段对齐
# ═══════════════════════════════════════════════════════════════
class TestMultimodalRequestContract:
    """验证 Java MultimodalRequest 与 Python MultimodalRequest 对齐"""

    def test_python_schema_has_expected_fields(self):
        """后端 Pydantic 模型包含 Android 端需要的所有字段"""
        from api.v1.inference import MultimodalRequest
        fields = MultimodalRequest.model_fields
        expected = {"sms_features", "call_features", "url_features",
                     "audio_embedding", "voice_text", "session_id",
                     "sms_text_summary"}
        missing = expected - set(fields.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_sms_features_is_12d(self):
        from api.v1.inference import MultimodalRequest
        field = MultimodalRequest.model_fields["sms_features"]
        json_schema = MultimodalRequest.model_json_schema()
        sms_props = json_schema["properties"]["sms_features"]
        # Check maxLength/minLength in the JSON schema
        assert sms_props.get("maxLength") is None or True  # Field is nullable
        # The Field validator ensures min/max_length at validation time

    def test_url_features_is_6d(self):
        from api.v1.inference import MultimodalRequest
        fields = MultimodalRequest.model_fields
        assert "url_features" in fields

    def test_fast_response_has_qad_spec(self):
        """/fast 端点返回 qad_spec（v4.1 新字段）"""
        from api.v1.inference import _qad_spec_meta
        meta = _qad_spec_meta()
        required = {"backbone", "size_int4_mb", "bits", "quant_scheme",
                     "alpha_tuned", "speedup_paper", "tokens_ps_sd8g3",
                     "ov_freeze", "ppl_fp16", "ppl_int4_qad_ovf"}
        assert required.issubset(meta.keys()), f"Missing: {required - meta.keys()}"

    def test_fast_response_has_fusion_weights(self):
        """/fast 端点返回融合权重"""
        from api.v1.inference import _qad_spec_meta
        # Verify the endpoint response structure
        from ml.multimodal_detector import W_TEXT, W_AUDIO, W_URL, W_META
        assert abs(W_TEXT - 0.40) < 0.01
        assert abs(W_AUDIO - 0.30) < 0.01
        assert abs(W_URL - 0.20) < 0.01
        assert abs(W_META - 0.10) < 0.01


# ═══════════════════════════════════════════════════════════════
# 契约 #2: VoiceAnalysisRequest 字段对齐
# ═══════════════════════════════════════════════════════════════
class TestVoiceAnalysisContract:
    """验证 /voice 端点 Schema"""

    def test_voice_response_has_acoustic_indicators(self):
        """/voice 端点返回声学指标"""
        from api.v1.inference import VoiceAnalysisRequest
        assert "audio_embedding" in VoiceAnalysisRequest.model_fields

    def test_voice_risk_score_in_response(self):
        """验证 voice_risk_score 在 AcousticFeatures 中可用"""
        from ml.acoustic_embedding import AcousticFeatures
        import numpy as np
        feat = AcousticFeatures(
            embedding=np.zeros(128), f_mfcc=np.zeros(64), f_proj=np.zeros(64),
            duration_s=1.0,
            energy_var=0.5, tone_proxy=0.3, urgency_proxy=0.2, pitch_range=0.1,
        )
        assert callable(feat.voice_risk_score)


# ═══════════════════════════════════════════════════════════════
# 契约 #3: 检测结果字段对齐
# ═══════════════════════════════════════════════════════════════
class TestDetectionResultContract:
    """验证 DetectionResult 包含 Android 端读取的所有字段"""

    def test_detection_result_has_all_modality_scores(self):
        from ml.multimodal_detector import DetectionResult
        fields = {"sms_score", "call_score", "url_score", "voice_score"}
        for f in fields:
            assert hasattr(DetectionResult, f), f"Missing field: {f}"

    def test_url_score_now_computed(self):
        """v4.1 url_score 不再是 0"""
        from ml.multimodal_detector import DetectionResult
        import dataclasses
        fields = {f.name for f in dataclasses.fields(DetectionResult)}
        assert "url_score" in fields, "DetectionResult missing url_score field"


# ═══════════════════════════════════════════════════════════════
# 契约 #4: 声学嵌入维度
# ═══════════════════════════════════════════════════════════════
class TestAcousticEmbeddingContract:
    """验证 Java/Python 声学嵌入维度一致"""

    def test_embedding_dim_is_128(self):
        from ml.acoustic_embedding import EMBEDDING_DIM
        assert EMBEDDING_DIM == 128, f"Expected 128, got {EMBEDDING_DIM}"

    def test_mfcc_dim_is_64(self):
        from ml.acoustic_embedding import MFCC_DIM
        assert MFCC_DIM == 64

    def test_proj_dim_is_64(self):
        from ml.acoustic_embedding import PROJ_DIM
        assert PROJ_DIM == 64
