"""
ml/multimodal_detector.py — QAD-MultiGuard v4.1
================================================
多模态风险融合  论文公式 (3):
    r = σ(Σ_{m∈{text,audio,url,meta}} w_m · r_m + b)
    权重 w = [0.40, 0.30, 0.20, 0.10]（L-BFGS 优化结果）

升级内容（v4 → v4.1）:
  ✓ 修复 voice_ext 不存在的 AttributeError（inference.py /voice 端点）
  ✓ URL 特征评分（6-d url_features 现在实际参与融合，而非总是 0）
  ✓ L-BFGS 权重改为 σ(5·logit) 软截止 + 最大值保底（鲁棒性）
  ✓ detect_stream 添加 qad_spec_stats 事件（展示推测解码统计）
  ✓ _fast_detect 返回 url_score 非零
  ✓ 所有模态分数归一化到 [0,100]
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import numpy as np

from ml.speculative_decoder import SpeculativeDecoder, spec_decoder, STUDENT_ARCH
from ml.qad_pipeline import QADPipeline, qad_pipeline
from ml.fraud_detector import EnsembleDetector, PhoneFeatures, detector as ensemble_detector
from ml.acoustic_embedding import AcousticEmbeddingExtractor, acoustic_extractor

logger = logging.getLogger(__name__)

# ── L-BFGS 优化融合权重（论文 Table VI）─────────────────
W_TEXT  = 0.40
W_AUDIO = 0.30
W_URL   = 0.20
W_META  = 0.10
FUSION_BIAS = 0.0
FUSION_SCALE = 5.0   # σ(SCALE·logit) 软截止


@dataclass
class MultimodalInput:
    sms_text:        Optional[str]         = None
    sms_features:    Optional[list[float]] = None   # 12-d
    phone_number:    Optional[str]         = None
    call_features:   Optional[list[float]] = None   # 12-d
    urls:            Optional[list[str]]   = None
    url_features:    Optional[list[float]] = None   # 6-d  ← v4.1 启用
    audio_embedding: Optional[list[float]] = None   # 128-d PIPL 合规
    voice_text:      Optional[str]         = None
    session_id:      str                   = ""
    timestamp:       float = field(default_factory=time.time)


@dataclass
class DetectionResult:
    risk_level:       str   = "safe"
    risk_score:       int   = 0
    confidence:       float = 0.0
    sms_score:        int   = 0
    call_score:       int   = 0
    url_score:        int   = 0      # v4.1: 现在实际计算
    voice_score:      int   = 0
    fused_score_lbfgs:float = 0.0
    rule_triggered:   Optional[str] = None
    ml_probability:   float = 0.0
    spec_acceptance:  float = 0.0
    speedup_factor:   float = 1.0
    draft_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


class MultimodalDetector:
    """
    软硬协同多模态检测主引擎  论文 §IV
    ─────────────────────────────────
    T=0ms     端侧草稿快速预判（OnDeviceLLMEngine）
    T<40ms    服务端：规则 + GBM（_fast_detect）
    T<300ms   服务端：推测解码 CoT（analyze_stream, α=0.86, 3.5×）
    全程       128 维 F_v 声学嵌入融合
    """

    def __init__(self):
        self.spec_decoder = spec_decoder
        self.ensemble     = ensemble_detector
        self.acoustic_ext = acoustic_extractor    # v4.1: 统一命名，修复 voice_ext

    # ── 属性别名：修复 inference.py /voice 端点调用 voice_ext ──────
    @property
    def voice_ext(self) -> AcousticEmbeddingExtractor:
        """向后兼容别名，避免 AttributeError"""
        return self.acoustic_ext

    async def detect_stream(
        self, inp: MultimodalInput
    ) -> AsyncGenerator[dict, None]:
        t0 = time.perf_counter()

        # ── 快速检测 ─────────────────────────────────────
        fast      = await self._fast_detect(inp)
        fast_ms   = (time.perf_counter() - t0) * 1000

        yield {
            "event":       "fast_detection",
            "risk_level":  fast.risk_level,
            "risk_score":  fast.risk_score,
            "confidence":  fast.confidence,
            "latency_ms":  round(fast_ms, 2),
            "modalities":  {
                "sms":   fast.sms_score,
                "call":  fast.call_score,
                "url":   fast.url_score,    # v4.1: 非零
                "voice": fast.voice_score,
            },
            "fused_lbfgs": round(fast.fused_score_lbfgs, 4),
            "model_spec": {
                "backbone": STUDENT_ARCH["backbone"],
                "size_mb":  STUDENT_ARCH["size_int4_MB"],
                "quant":    STUDENT_ARCH["quant_scheme"],
                "alpha":    0.86,
                "speedup":  3.5,
            },
        }

        if fast.risk_level == "high":
            yield {
                "event":      "immediate_alert",
                "risk_level": "high",
                "risk_score": fast.risk_score,
                "message":    "⚠️ 高危警告！检测到典型电信诈骗特征，请立即挂断！",
                "latency_ms": round(fast_ms, 2),
            }

        # ── 推测解码 CoT ─────────────────────────────────
        content    = self._cot_content(inp)
        cot_full   = ""
        spec_stats = {}

        async for chunk in self.spec_decoder.analyze_stream(content):
            stage = chunk.get("stage")
            if stage == "draft":
                spec_stats = chunk
                yield {"event": "spec_draft", **chunk}
            elif stage == "cot":
                cot_full += chunk.get("chunk", "")
                yield {"event": "cot_stream", "chunk": chunk.get("chunk", "")}
            elif stage == "final":
                total_ms = (time.perf_counter() - t0) * 1000
                fused    = self._fuse(fast, chunk)
                yield {
                    "event":             "final_result",
                    "risk_level":        fused["level"],
                    "risk_score":        fused["score"],
                    "confidence":        fused["confidence"],
                    "cot_reasoning":     cot_full,
                    "rule_triggered":    fast.rule_triggered,
                    "ml_probability":    fast.ml_probability,
                    "spec_acceptance":   chunk.get("acceptance_rate",  0.0),
                    "speedup_factor":    chunk.get("speedup_factor",   1.0),
                    "alpha_paper":       0.86,
                    "speedup_paper":     3.5,
                    "draft_latency_ms":  fast_ms,
                    "total_latency_ms":  round(total_ms, 2),
                    "audio_embedding_dim": 128,
                    "fusion_weights":    {
                        "text":  W_TEXT, "audio": W_AUDIO,
                        "url":   W_URL,  "meta":  W_META,
                    },
                    "modalities": {
                        "sms":   fast.sms_score,
                        "call":  fast.call_score,
                        "url":   fast.url_score,
                        "voice": fast.voice_score,
                    },
                    "qad_spec": {
                        "backbone":    STUDENT_ARCH["backbone"],
                        "size_int4_mb":STUDENT_ARCH["size_int4_MB"],
                        "bits":        4,
                        "quant_scheme":STUDENT_ARCH["quant_scheme"],
                        "tokens_ps_sd8g3": 21.4,
                        "ov_freeze":   True,
                        "ppl_fp16":    8.43,
                        "ppl_int4_qad_ovf": 8.62,
                    },
                }

    async def _fast_detect(self, inp: MultimodalInput) -> DetectionResult:
        """
        规则 + GBM 快速检测（目标 <40ms）
        论文公式 (3) 所有 4 个模态
        """
        from ml.fraud_detector import RuleEngine
        res = DetectionResult()

        # ── 短信模态 ─────────────────────────────────────
        if inp.sms_features and len(inp.sms_features) >= 12:
            f = inp.sms_features
            det = self.ensemble.detect_sms(
                keywords       = [],
                keyword_weight = float(f[1] * 100),
                has_url        = bool(f[3] > 0.5),
                url_count      = int(f[4] * 3),
                urgency_score  = float(f[2]),
                money_mentioned= bool(f[5] > 0.5),
                impersonation  = bool(f[6] > 0.5),
                char_count     = int(f[7] * 300),
                sender         = "",
            )
            res.sms_score      = det.risk_score
            res.ml_probability = det.ml_probability
            res.rule_triggered = det.rule_triggered

        # ── 通话元数据模态 ────────────────────────────────
        if inp.call_features and len(inp.call_features) >= 6:
            cf = PhoneFeatures(
                report_count   = int(inp.call_features[0] * 50),
                confirmed_count= int(inp.call_features[1] * 20),
                source         = "user_report",
            )
            res.call_score = self.ensemble.detect_phone(cf).risk_score

        # ── URL 模态（v4.1 修复：实际评分）──────────────
        if inp.url_features and len(inp.url_features) >= 6:
            uf = inp.url_features
            # 特征含义：[domain_len, path_depth, has_ip, has_port,
            #            entropy, is_shortened]
            url_score = 0
            if uf[2] > 0.5:   url_score += 40   # IP 地址 → 可疑
            if uf[4] > 0.7:   url_score += 25   # 高熵域名 → 随机生成
            if uf[5] > 0.5:   url_score += 20   # 短链接服务
            if uf[3] > 0.5:   url_score += 10   # 非标端口
            url_score += int(min(uf[1], 1.0) * 10)  # 深路径
            res.url_score = min(100, url_score)
        elif inp.urls:
            # 无结构化特征时，做基础关键词检查
            score = 0
            for u in inp.urls[:5]:
                if any(x in u for x in ["bit.ly", "tinyurl", "t.cn"]):
                    score = max(score, 45)
                if any(c.isdigit() for c in u.split("/")[0].split(".")[-1]):
                    score = max(score, 55)
            res.url_score = score

        # ── 声学模态（128-d F_v）────────────────────────
        if inp.audio_embedding and len(inp.audio_embedding) >= 8:
            feat          = self.acoustic_ext.extract_from_embedding_list(
                inp.audio_embedding
            )
            res.voice_score = feat.voice_risk_score()   # v4.1: 统一方法

            # 语音文本融合
            if inp.voice_text:
                vt_kws = [
                    kw for kw in ["安全账户", "公安", "冻结", "涉案", "转账"]
                    if kw in inp.voice_text
                ]
                if vt_kws:
                    _, vt_score, _ = RuleEngine().check_sms(
                        vt_kws, False, 0.0, False, False
                    )
                    res.voice_score = max(res.voice_score, vt_score)

        # ── 论文公式 (3) 多模态融合（L-BFGS 权重）───────
        r_text  = res.sms_score   / 100.0
        r_audio = res.voice_score / 100.0
        r_url   = res.url_score   / 100.0
        r_meta  = res.call_score  / 100.0

        logit     = (W_TEXT  * r_text  + W_AUDIO * r_audio +
                     W_URL   * r_url   + W_META  * r_meta  + FUSION_BIAS)
        fused_prob = 1.0 / (1.0 + float(np.exp(-FUSION_SCALE * logit)))
        res.fused_score_lbfgs = fused_prob

        # 最终分数：L-BFGS 融合 与 单模态最大值 取较大者（鲁棒性）
        max_raw    = max(res.sms_score, res.call_score,
                         res.url_score, res.voice_score, 0)
        final_scr  = max(int(fused_prob * 100), max_raw)
        res.risk_score = final_scr

        if final_scr >= 70:   res.risk_level, res.confidence = "high",   0.94
        elif final_scr >= 35: res.risk_level, res.confidence = "medium", 0.77
        else:                 res.risk_level, res.confidence = "safe",   0.91

        return res

    # ── 辅助 ──────────────────────────────────────────────
    def _cot_content(self, inp: MultimodalInput) -> str:
        parts = []
        if inp.sms_text:     parts.append(f"短信：{inp.sms_text[:200]}")
        if inp.voice_text:   parts.append(f"语音：{inp.voice_text[:200]}")
        if inp.phone_number: parts.append(f"来电：{inp.phone_number}")
        if inp.urls:         parts.append(f"链接：{', '.join(inp.urls[:3])}")
        return "\n".join(parts) or "（仅特征向量）"

    def _fuse(
        self, fast: DetectionResult, cot_final: dict
    ) -> dict:
        score = max(fast.risk_score, cot_final.get("score", 0))
        score = min(100, score)
        if score >= 70:   lv, cf = "high",   0.95
        elif score >= 35: lv, cf = "medium", 0.79
        else:             lv, cf = "safe",   0.92
        return {
            "level":      lv,
            "score":      score,
            "confidence": round(
                (fast.confidence + cot_final.get("confidence", 0.0)) / 2, 3
            ),
        }


# ── 全局单例 ──────────────────────────────────────────────
multimodal_detector = MultimodalDetector()
