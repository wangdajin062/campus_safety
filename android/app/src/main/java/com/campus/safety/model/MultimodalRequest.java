package com.campus.safety.model;

import java.util.List;

/**
 * MultimodalRequest — QAD-MultiGuard v4.1
 * =========================================
 * 对应后端 /v1/infer/stream 和 /v1/infer/fast 请求体
 *
 * 字段完全对齐后端 MultimodalRequest Pydantic Schema：
 *   sms_features(12d) + call_features(12d) + url_features(6d)
 *   + audio_embedding(128d, F_v) + voice_text + session_id
 *
 * 隐私保证（PIPL §23）：
 *   - sms_text 原文 不上传，只上传端侧提取的 12 维特征向量
 *   - audio 原始音频 不上传，只上传 F_v = [f_mfcc(64);W_proj·h̄_w(64)]
 */
public class MultimodalRequest {

    // ── 短信模态（12 维特征向量，端侧提取）────────────────
    /**
     * 12 维 SMS 特征向量
     * [keyword_hits/5, keyword_weight/100, urgency_score,
     *  has_url, url_count/3, money_mentioned, impersonation,
     *  char_count/300, digit_ratio, sender_is_number,
     *  kw_and_url, impersonation_and_money]
     */
    public List<Float> sms_features;

    /**
     * 脱敏短信摘要（可选，提升 CoT 推理质量）
     * 注意：不得包含手机号/姓名等个人信息
     */
    public String sms_text_summary;

    // ── 通话模态（12 维特征向量）──────────────────────────
    /**
     * 12 维通话行为特征向量
     * [report_count/50, confirmed_count/20, query_count/100,
     *  days_since_first/365, source_score, location_flag,
     *  carrier_flag, high_report_flag, confirmed_flag,
     *  report_no_confirm, log_report/6, police_flag]
     */
    public List<Float> call_features;

    /** 来电号码（可选，用于 CoT 内容） */
    public String phone_number;

    // ── URL 模态（6 维特征向量）───────────────────────────
    /**
     * 6 维 URL 结构特征
     * [domain_len/100, path_depth/10, has_ip_as_domain,
     *  has_non_std_port, entropy/5.0, is_shortened_service]
     *
     * v4.1 新增：现在实际参与 L-BFGS 融合（w_url=0.20）
     */
    public List<Float> url_features;

    // ── 声学模态（128 维隐私保护嵌入）────────────────────
    /**
     * 128 维声学特征嵌入 F_v = [f_mfcc(64d) ; W_proj·h̄_w(64d)]
     * 由 SmsFeatureExtractor / OnDeviceLLMEngine 端侧计算
     * 原始音频永远不上传（PIPL §23 合规）
     */
    public List<Float> audio_embedding;

    /** 端侧语音转文字结果（Whisper Tiny，可选） */
    public String voice_text;

    /** 会话 ID（用于日志追踪） */
    public String session_id = "";

    /** 是否启用 CoT 推理（true → 使用 /stream，false → 使用 /fast） */
    public boolean enable_cot = true;

    // ─────────────────────────────────────────────────────
    //  建造者模式
    // ─────────────────────────────────────────────────────

    public static Builder builder() { return new Builder(); }

    public static class Builder {
        private final MultimodalRequest req = new MultimodalRequest();

        public Builder smsFeatures(List<Float> v)   { req.sms_features = v;   return this; }
        public Builder smsSummary(String s)          { req.sms_text_summary = s; return this; }
        public Builder callFeatures(List<Float> v)  { req.call_features = v;  return this; }
        public Builder phoneNumber(String p)         { req.phone_number = p;   return this; }
        public Builder urlFeatures(List<Float> v)   { req.url_features = v;   return this; }
        public Builder audioEmbedding(List<Float> v){ req.audio_embedding = v; return this; }
        public Builder voiceText(String t)           { req.voice_text = t;     return this; }
        public Builder sessionId(String id)          { req.session_id = id;    return this; }
        public Builder enableCot(boolean b)          { req.enable_cot = b;     return this; }

        public MultimodalRequest build() { return req; }
    }

    /**
     * 检查是否有任何有效输入
     */
    public boolean hasAnyInput() {
        return (sms_features   != null && !sms_features.isEmpty())   ||
               (call_features  != null && !call_features.isEmpty())  ||
               (url_features   != null && !url_features.isEmpty())   ||
               (audio_embedding != null && !audio_embedding.isEmpty());
    }

    /**
     * 活跃模态数量（用于 UI 展示）
     */
    public int activeModalityCount() {
        int n = 0;
        if (sms_features   != null && !sms_features.isEmpty())   n++;
        if (call_features  != null && !call_features.isEmpty())  n++;
        if (url_features   != null && !url_features.isEmpty())   n++;
        if (audio_embedding!= null && !audio_embedding.isEmpty()) n++;
        return n;
    }
}
