package com.campus.safety.model;

import java.util.List;
import java.util.Map;

/**
 * CoTStreamEvent — QAD-MultiGuard v4.1
 * =====================================
 * 对应后端 /v1/infer/stream SSE 事件数据模型
 *
 * 升级（v4 → v4.1）：
 *   ✓ 添加 qad_spec 字段（推测解码规格）
 *   ✓ 添加 fused_lbfgs（论文公式3融合分数）
 *   ✓ 添加 fusion_weights（4模态权重）
 *   ✓ 添加 modalities 内嵌类（各模态分数）
 *   ✓ url_score 现在有意义的值（非零）
 *   ✓ acoustic_indicators 字段（韵律检测）
 *
 * 对应后端 SSE event types：
 *   fast_detection | immediate_alert | spec_draft
 *   cot_stream | final_result | error | done
 */
public class CoTStreamEvent {

    // ── 通用 ─────────────────────────────────────────────
    public String event;          // 事件类型（有时 Gson 从字段解析）
    public String risk_level;     // "safe" | "medium" | "high"
    public int    risk_score;     // [0, 100]
    public float  confidence;     // [0.0, 1.0]
    public float  latency_ms;
    public String message;        // immediate_alert 提示文字
    public boolean done;
    public String error;

    // ── fast_detection / final_result ────────────────────
    public Modalities modalities;
    public float      fused_lbfgs;   // 论文公式(3) L-BFGS 融合分数
    public FusionWeights fusion_weights;
    public QadSpec    qad_spec;
    public ModelSpec  model_spec;

    // ── spec_draft ────────────────────────────────────────
    /** 推测解码草稿事件 */
    public int    tokens_accepted;
    public int    tokens_total;
    public float  alpha;             // 实测接受率
    public int    gamma;             // 草稿 token 数
    public float  acceptance_rate;
    public float  speedup_factor;
    public String model_type;        // 学生模型名称

    // ── cot_stream ────────────────────────────────────────
    public String chunk;             // CoT 推理 token 流

    // ── final_result ─────────────────────────────────────
    public String  cot_reasoning;
    public String  rule_triggered;
    public float   ml_probability;
    public float   spec_acceptance;
    public float   draft_latency_ms;
    public float   total_latency_ms;
    public int     audio_embedding_dim;

    // ── /voice 端点专用 ───────────────────────────────────
    public AcousticIndicators acoustic_indicators;
    public int    voice_score;
    public int    text_score;
    public float  call_duration_s;

    // ── /fast 端点额外字段 ────────────────────────────────
    public QadSpec qad_spec_fast;    // fast 端点返回的 qad_spec

    // ─────────────────────────────────────────────────────
    //  内嵌类
    // ─────────────────────────────────────────────────────

    public static class Modalities {
        public int sms;
        public int call;
        public int url;     // v4.1: 现在有意义（URL 特征评分）
        public int voice;
    }

    public static class FusionWeights {
        public float text;   // 0.40
        public float audio;  // 0.30
        public float url;    // 0.20
        public float meta;   // 0.10
    }

    /**
     * QAD 推测解码规格（论文 Table I + §V）
     * 附在 fast_detection / final_result 事件中
     */
    public static class QadSpec {
        public String backbone;         // "Qwen2.5-0.5B-Instruct"
        public int    size_int4_mb;     // 240
        public int    bits;             // 4
        public String quant_scheme;     // "Q4_K_M"
        public float  alpha_tuned;      // 0.86
        public float  speedup_paper;    // 3.5
        public float  tokens_ps_sd8g3;  // 21.4 tok/s
        public boolean ov_freeze;       // OV-Freeze 策略
        public float  ppl_fp16;         // 8.43
        public float  ppl_int4_qad_ovf; // 8.62
        public int    hidden_dim;       // 896
        public int    n_layers;         // 24
    }

    /** spec_draft 事件中的 model_spec */
    public static class ModelSpec {
        public String backbone;
        public int    size_mb;
        public int    bits;
        public String scheme;
    }

    /** /voice 端点返回的韵律指标 */
    public static class AcousticIndicators {
        public float energy_variance;  // 能量方差（高→语速不均）
        public float tone_proxy;       // 音调代理（高→疑似模仿官方）
        public float urgency_proxy;    // 紧迫感代理
        public float pitch_range;      // 音高范围
        public int   voice_risk_score; // [0,100]
        public float duration_s;       // 音频时长
        public Float dp_epsilon;       // DP 隐私参数（null=关闭）
    }

    // ─────────────────────────────────────────────────────
    //  便捷方法
    // ─────────────────────────────────────────────────────

    /** 是否高危 */
    public boolean isHighRisk() {
        return "high".equals(risk_level);
    }

    /** 是否需要展示推测解码统计卡 */
    public boolean hasSpecStats() {
        return tokens_total > 0 || alpha > 0;
    }

    /** 获取展示用 QAD 规格文本 */
    public String getQadSummary() {
        QadSpec spec = qad_spec != null ? qad_spec : qad_spec_fast;
        if (spec == null) return "QAD-MG (Qwen2.5-0.5B · INT4 · 240MB)";
        return String.format("%s · %s · %dMB · α=%.2f · %.1f tok/s",
            spec.backbone, spec.quant_scheme, spec.size_int4_mb,
            spec.alpha_tuned, spec.tokens_ps_sd8g3);
    }

    /** 获取融合权重描述 */
    public String getFusionWeightsText() {
        if (fusion_weights == null) return "text:40% audio:30% url:20% meta:10%";
        return String.format("text:%.0f%% audio:%.0f%% url:%.0f%% meta:%.0f%%",
            fusion_weights.text  * 100, fusion_weights.audio * 100,
            fusion_weights.url   * 100, fusion_weights.meta  * 100);
    }
}
