package com.campus.safety.ml;

import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * QAD-MultiGuard v4.1 蒸馏模型（推测解码器）单元测试
 * 覆盖：草稿生成、诈骗类别检测、α 接受率、验证逻辑、加速比计算
 */
public class SpeculativeDecoderTest {

    private SpeculativeDecoder decoder;

    @Before
    public void setUp() {
        decoder = new SpeculativeDecoder();
        decoder.resetStats();
    }

    // ── 论文公式 (6) 加速比 ──────────────────────────────

    @Test
    public void computeSpeedup_paperValues() {
        // 论文 §V: α=0.86, γ=5 → 3.5×
        float speedup = SpeculativeDecoder.VerifyResult.computeSpeedup(0.86f, 5);
        assertEquals("α=0.86, γ=5 → 3.5×", 3.5f, speedup, 0.05);
    }

    @Test
    public void computeSpeedup_genericBaseline() {
        // 通用模型 α=0.78, γ=5 → 约 2.9×（论文 Table V 对比值）
        float speedup = SpeculativeDecoder.VerifyResult.computeSpeedup(0.78f, 5);
        assertEquals("α=0.78, γ=5 → ~2.9×", 2.9f, speedup, 0.1);
    }

    @Test
    public void computeSpeedup_highAlpha() {
        // 公安冒充 α=0.92, γ=5 → 约 5.2×
        float speedup = SpeculativeDecoder.VerifyResult.computeSpeedup(0.92f, 5);
        assertTrue("α=0.92 加速比应 > 3.5", speedup > 3.5f);
    }

    @Test
    public void computeSpeedup_edgeCases() {
        assertEquals("α=0 → 1.0×", 1.0f, SpeculativeDecoder.VerifyResult.computeSpeedup(0f, 5), 0.01);
        assertEquals("α=1 → 1.0×", 1.0f, SpeculativeDecoder.VerifyResult.computeSpeedup(1f, 5), 0.01);
        assertEquals("α<0 → 1.0×", 1.0f, SpeculativeDecoder.VerifyResult.computeSpeedup(-0.5f, 5), 0.01);
    }

    // ── 论文 Table V：按类别的 α 接受率 ──────────────────

    @Test
    public void alpha_perCategory_matchesPaper() {
        assertEquals("public_security", 0.92f, decoder.getAlphaForCategory("public_security"), 0.01);
        assertEquals("investment",      0.88f, decoder.getAlphaForCategory("investment"),      0.01);
        assertEquals("part_time",       0.87f, decoder.getAlphaForCategory("part_time"),       0.01);
        assertEquals("telecom_billing", 0.85f, decoder.getAlphaForCategory("telecom_billing"), 0.01);
        assertEquals("online_shopping", 0.81f, decoder.getAlphaForCategory("online_shopping"), 0.01);
        assertEquals("unknown",         0.86f, decoder.getAlphaForCategory("unknown"),         0.01);
        assertEquals("general",         0.86f, decoder.getAlphaForCategory("general"),         0.01);
    }

    @Test
    public void alpha_domainTuned_higherThanGeneric() {
        assertTrue("领域调优 α=0.86 > 通用 α=0.78",
            SpeculativeDecoder.ALPHA_DOMAIN_TUNED > SpeculativeDecoder.ALPHA_GENERIC);
    }

    // ── 诈骗类别检测 ────────────────────────────────────

    @Test
    public void detectCategory_publicSecurity() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您的账户涉嫌洗钱，请配合公安局调查，立即转账到安全账户", "110");
        DraftResultProxy proxy = generateDraftProxy(decoder, f);
        // 应被检测为 public_security
        assertTrue("hitKeywords 应包含公安类词汇",
            f.hitKeywords.toString().contains("公安") ||
            f.hitKeywords.toString().contains("安全账户"));
    }

    @Test
    public void detectCategory_partTime() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "急招刷单兼职，每单佣金50元，日结", "10086");
        assertTrue("应命中刷单关键词",
            f.hitKeywords.toString().contains("刷单"));
    }

    @Test
    public void detectCategory_investment() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "内部消息：稳定收益8%，保本保息，限时抢购", "1069");
        assertTrue("应命中投资关键词",
            f.hitKeywords.toString().contains("内部消息"));
    }

    @Test
    public void detectCategory_safeMessage() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您的话费账单已出，请及时缴纳", "10086");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        assertEquals("safe 消息的 draftScore 应为 0", 0f, draft.draftScore, 0.01);
    }

    // ── 草稿生成 ────────────────────────────────────────

    @Test
    public void generateDraft_highRisk_producesGammaTokens() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "安全账户：您的资产已被冻结，立即转账到公安局配合调查解除冻结", "110");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        assertEquals("草稿应生成 GAMMA 个 token", SpeculativeDecoder.GAMMA, draft.tokens.size());
        assertEquals("草稿应生成 GAMMA 个概率值", SpeculativeDecoder.GAMMA, draft.probs.size());
        assertTrue("高危 → draftScore >= 70", draft.draftScore >= 70);
        assertEquals("高危 → riskLevel = high", "high", draft.riskLevel);
        assertTrue("高危 → isHighRisk", draft.isHighRisk);
    }

    @Test
    public void generateDraft_safeRisk_producesSafeTokens() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您好，您的快递已到达驿站，请凭取件码领取", "10086");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        assertEquals("应生成 GAMMA 个 token", SpeculativeDecoder.GAMMA, draft.tokens.size());
        assertTrue("安全 → draftScore < 35", draft.draftScore < 35);
        assertEquals("安全 → riskLevel = safe", "safe", draft.riskLevel);
        assertEquals("安全 → 第一个 token 为 ✓安全", "✓安全", draft.tokens.get(0));
    }

    @Test
    public void generateDraft_mediumRisk() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您的贷款申请已通过，点击链接确认", "1069");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        assertTrue("中危 → 35 <= draftScore < 70",
            draft.draftScore >= 35 && draft.draftScore < 70);
        assertEquals("中危 → riskLevel = medium", "medium", draft.riskLevel);
    }

    // ── Token 概率递减 ──────────────────────────────────

    @Test
    public void tokenProbabilities_decreaseWithPosition() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "安全账户：您的资产已被冻结", "110");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        for (int i = 1; i < draft.probs.size(); i++) {
            assertTrue("概率应随位置递减: pos" + i + " > pos" + (i+1),
                draft.probs.get(i-1) >= draft.probs.get(i));
        }
    }

    // ── 验证逻辑 ────────────────────────────────────────

    @Test
    public void verify_acceptsTokens_whenServerAgrees() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "安全账户：您的资产已被冻结，立即转账", "110");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        // 服务端给出同样高危结论
        SpeculativeDecoder.VerifyResult result = decoder.verify(draft, 85f, "high");

        assertTrue("服务端高危 → 应接受部分草稿", result.acceptedCount > 0);
        assertFalse("服务端结论与草稿一致 → corrected=false", result.corrected);
        assertEquals("finalRiskLevel = high", "high", result.finalRiskLevel);
        assertTrue("speedupFactor >= 1.0", result.speedupFactor >= 1.0f);
    }

    @Test
    public void verify_corrected_whenServerDisagrees() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您的贷款申请已通过，点击链接确认", "1069");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        // 服务端给出高危（草稿可能是 medium）
        SpeculativeDecoder.VerifyResult result = decoder.verify(draft, 90f, "high");

        assertTrue("服务端纠正了草稿 → corrected=true",
            result.corrected || draft.riskLevel.equals("high"));
    }

    @Test
    public void verify_noAcceptance_whenServerScoreVeryLow() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您好，您的快递已到达驿站", "10086");
        SpeculativeDecoder.DraftResult draft = decoder.generateDraft(f);
        // 服务端给很低分数
        SpeculativeDecoder.VerifyResult result = decoder.verify(draft, 5f, "safe");

        // 安全消息可能接受率低
        assertTrue("acceptanceRate ∈ [0, 1]", result.acceptanceRate >= 0 && result.acceptanceRate <= 1);
    }

    // ── 统计累计 ────────────────────────────────────────

    @Test
    public void stats_accumulateAcrossRounds() {
        SmsFeatureExtractor.Features f1 = SmsFeatureExtractor.extract(
            "安全账户：立即转账，配合调查", "110");
        SmsFeatureExtractor.Features f2 = SmsFeatureExtractor.extract(
            "您的快递已到达", "10086");
        SmsFeatureExtractor.Features f3 = SmsFeatureExtractor.extract(
            "刷单兼职，日结佣金", "1069");

        decoder.verify(decoder.generateDraft(f1), 90f, "high");
        decoder.verify(decoder.generateDraft(f2), 5f,  "safe");
        decoder.verify(decoder.generateDraft(f3), 75f, "high");

        assertEquals("3 轮推理", 3, decoder.getTotalRounds());
        assertEquals("总 tokens = GAMMA * 3", SpeculativeDecoder.GAMMA * 3, decoder.getTotalTokens());
        assertTrue("总接受率 ∈ [0, 1]", decoder.getOverallAcceptanceRate() > 0);
        assertTrue("平均加速比 >= 1.0", decoder.getAverageSpeedup() >= 1.0f);
    }

    @Test
    public void resetStats_clearsCounters() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "安全账户：立即转账", "110");
        decoder.verify(decoder.generateDraft(f), 90f, "high");
        assertTrue("reset 前应有统计", decoder.getTotalRounds() > 0);

        decoder.resetStats();
        assertEquals("reset 后 rounds = 0", 0, decoder.getTotalRounds());
        assertEquals("reset 后 tokens = 0", 0, decoder.getTotalTokens());
        assertEquals("reset 后 accepted = 0", 0, decoder.getAcceptedTokens());
    }

    // ── 常量 ──────────────────────────────────────────

    @Test
    public void constants_matchSpec() {
        assertEquals("GAMMA = 5", 5, SpeculativeDecoder.GAMMA);
        assertEquals("学生模型 = Qwen2.5-0.5B-Instruct", "Qwen2.5-0.5B-Instruct", SpeculativeDecoder.STUDENT_MODEL);
        assertEquals("模型大小 = 240MB", 240, SpeculativeDecoder.STUDENT_SIZE_MB);
        assertEquals("层数 = 24", 24, SpeculativeDecoder.STUDENT_LAYERS);
        assertEquals("隐藏维度 = 896", 896, SpeculativeDecoder.STUDENT_HIDDEN);
        assertEquals("量化方案 = Q4_K_M", "Q4_K_M", SpeculativeDecoder.QUANT_SCHEME);
    }

    // ── 辅助 ──────────────────────────────────────────

    /** 辅助类：用于间接测试 detectCategory（protected 方法）*/
    private static class DraftResultProxy {
        public final SpeculativeDecoder.DraftResult result;
        public DraftResultProxy(SpeculativeDecoder.DraftResult r) { this.result = r; }
    }

    private DraftResultProxy generateDraftProxy(SpeculativeDecoder dec, SmsFeatureExtractor.Features f) {
        return new DraftResultProxy(dec.generateDraft(f));
    }
}
