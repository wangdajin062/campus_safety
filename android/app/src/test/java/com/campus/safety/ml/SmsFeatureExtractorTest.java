package com.campus.safety.ml;

import com.campus.safety.model.MultimodalRequest;

import org.junit.Test;
import static org.junit.Assert.*;

import java.util.Arrays;
import java.util.List;

/**
 * SmsFeatureExtractor 特征提取单元测试
 * 覆盖：12 维 SMS 特征、6 维 URL 特征、12 维来电特征、Features 端侧评分、buildRequest 构建
 */
public class SmsFeatureExtractorTest {

    private final SmsFeatureExtractor extractor = new SmsFeatureExtractor();

    // ── Features 端侧提取 ─────────────────────────────

    @Test
    public void extract_safeMessage() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您好，您的话费账单已出，请及时通过官方渠道缴纳", "10086");
        assertEquals("无关键词命中", 0, f.hitKeywords.size());
        assertFalse("无 URL", f.hasUrl);
        assertEquals("score = 0", 0, f.quickScore());
        assertEquals("riskLevel = safe", "safe", f.quickRiskLevel());
    }

    @Test
    public void extract_publicSecurityHighRisk() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "安全账户：您的资产已被冻结，请配合公安局调查，立即转账到安全账户解冻", "110");
        assertTrue("命中公安类关键词", f.hitKeywords.size() >= 3);
        assertTrue("评分 >= 70", f.quickScore() >= 70);
        assertEquals("风险等级 = high", "high", f.quickRiskLevel());
        assertTrue("涉钱标记", f.moneyMentioned);
        assertTrue("冒充标记", f.impersonation);
    }

    @Test
    public void extract_withUrl() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "您的账户异常，请立即点击 https://bit.ly/abc123 验证", "1069");
        assertTrue("应检测到 URL", f.hasUrl);
        assertEquals("URL 计数", 1, f.urlCount);
    }

    @Test
    public void extract_urgencyDetected() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "紧急！您的账户已被冻结！立即处理！否则将永久封禁！！", "10086");
        assertTrue("紧迫感分值 > 0", f.urgencyScore > 0);
    }

    @Test
    public void extract_ruleTriggered() {
        SmsFeatureExtractor.Features f = SmsFeatureExtractor.extract(
            "安全账户：您的资产冻结，请配合调查，立即转账到安全账户解冻", "110");
        assertNotNull("多个高危关键词 → 触发规则", f.ruleTriggered);
    }

    // ── 12 维 SMS 向量 ───────────────────────────────

    @Test
    public void buildSmsFeatures_safeMessage() {
        List<Float> v = extractor.buildSmsFeatures("您好，话费账单已出", "10086");
        assertEquals("维度 = 12", 12, v.size());
        assertEquals("keyword_hits = 0", 0.0f, v.get(0), 0.01);
        assertEquals("keyword_weight = 0", 0.0f, v.get(1), 0.01);
        assertEquals("has_url = 0", 0.0f, v.get(2), 0.01);
        assertEquals("sender_is_number = 1 (10086)", 1.0f, v.get(9), 0.01);
    }

    @Test
    public void buildSmsFeatures_fraudMessage() {
        List<Float> v = extractor.buildSmsFeatures(
            "安全账户：您的资产已被冻结，立即转账到安全账户，配合调查", "未知号码");
        assertTrue("keyword_hits > 0", v.get(0) > 0);
        assertTrue("keyword_weight > 0", v.get(1) > 0);
        assertEquals("sender_is_number = 0", 0.0f, v.get(9), 0.01);
        assertEquals("keyword_hits>0 AND has_url feature = 0", 0.0f, v.get(10), 0.01);
    }

    @Test
    public void buildSmsFeatures_withUrlAndMoney() {
        List<Float> v = extractor.buildSmsFeatures(
            "恭喜中奖！点击 http://t.cn/xyz 领取50万元奖金", "1069");
        assertEquals("has_url = 1", 1.0f, v.get(3), 0.01);
        assertEquals("money_mentioned = 1", 1.0f, v.get(5), 0.01);
        assertEquals("url_count > 0", 1.0f, v.get(4), 0.01);
    }

    @Test
    public void buildSmsFeatures_impersonation() {
        List<Float> v = extractor.buildSmsFeatures(
            "公安局：您涉及一起洗钱案件，请配合调查转账", "110");
        assertEquals("impersonation = 1", 1.0f, v.get(6), 0.01);
    }

    @Test
    public void buildSmsFeatures_urgency() {
        List<Float> v = extractor.buildSmsFeatures(
            "立即！马上！！紧急！！！", "10086");
        assertTrue("urgency_score > 0", v.get(2) > 0);
    }

    // ── 6 维 URL 特征 ───────────────────────────────

    @Test
    public void buildUrlFeatures_null() {
        assertNull("null urls → null", extractor.buildUrlFeatures(null));
        assertNull("empty urls → null", extractor.buildUrlFeatures(Arrays.asList()));
    }

    @Test
    public void buildUrlFeatures_normalUrl() {
        List<Float> v = extractor.buildUrlFeatures(
            Arrays.asList("https://example.com/path/page.html"));
        assertEquals("维度 = 6", 6, v.size());
        assertEquals("domain_len > 0", true, v.get(0) > 0);
        assertEquals("has_ip = 0", 0.0f, v.get(2), 0.01);
        assertEquals("is_shortened = 0", 0.0f, v.get(5), 0.01);
    }

    @Test
    public void buildUrlFeatures_ipDomain() {
        List<Float> v = extractor.buildUrlFeatures(
            Arrays.asList("http://192.168.1.1/admin/login"));
        assertEquals("has_ip = 1 (IP 地址域名)", 1.0f, v.get(2), 0.01);
    }

    @Test
    public void buildUrlFeatures_shortened() {
        List<Float> v = extractor.buildUrlFeatures(
            Arrays.asList("https://bit.ly/abc123"));
        assertEquals("is_shortened = 1", 1.0f, v.get(5), 0.01);
    }

    @Test
    public void buildUrlFeatures_multipleUrls() {
        List<Float> v = extractor.buildUrlFeatures(Arrays.asList(
            "https://example.com/path1",
            "https://bit.ly/xyz",
            "http://192.168.1.1"
        ));
        assertTrue("has_ip = 1 (第三个是 IP)", v.get(2) > 0);
        assertTrue("is_shortened = 1 (第二个是短链接)", v.get(5) > 0);
    }

    // ── 12 维来电特征 ───────────────────────────────

    @Test
    public void buildCallFeatures_normal() {
        List<Float> v = extractor.buildCallFeatures(0, 0, 0, 0f, 0f);
        assertEquals("维度 = 12", 12, v.size());
        assertEquals("无举报 → report=0", 0.0f, v.get(0), 0.01);
        assertEquals("无确认 → confirm=0", 0.0f, v.get(1), 0.01);
        assertEquals("未确认 → reportNoConfirm=0", 0.0f, v.get(8), 0.01);
    }

    @Test
    public void buildCallFeatures_highReport() {
        List<Float> v = extractor.buildCallFeatures(50, 10, 200, 180f, 0.6f);
        assertEquals("report=1.0 (capped)", 1.0f, v.get(0), 0.01);
        assertEquals("confirmed=0.5", 0.5f, v.get(1), 0.01);
        assertEquals("query=1.0 (capped)", 1.0f, v.get(2), 0.01);
        assertEquals("highReport=1", 1.0f, v.get(7), 0.01);
        assertEquals("confirmed=1", 1.0f, v.get(8), 0.01);
    }

    @Test
    public void buildCallFeatures_policeSource() {
        List<Float> v = extractor.buildCallFeatures(5, 0, 10, 30f, 1.0f);
        assertEquals("police_flag=1", 1.0f, v.get(11), 0.01);
        assertEquals("sourceScore=1.0", 1.0f, v.get(4), 0.01);
    }

    // ── buildRequest 完整构建 ────────────────────────

    @Test
    public void buildRequest_constructsCompleteRequest() {
        String content = "安全账户：您的账户冻结，请立即转账";
        String sender  = "110";
        List<String> urls = null;
        List<Float> audioEmb = Arrays.asList(new Float[128]);
        Arrays.fill(audioEmb.toArray(), 0.5f);

        MultimodalRequest req = extractor.buildRequest(content, sender, urls, audioEmb, "session-001");
        assertNotNull("request 不为 null", req);
        assertEquals("session_id", "session-001", req.session_id);
    }

    @Test
    public void buildRequest_withUrls() {
        MultimodalRequest req = extractor.buildRequest(
            "点击 http://bit.ly/abc 领奖", "1069",
            Arrays.asList("http://bit.ly/abc"),
            null, "session-002");
        assertNotNull("request 不为 null", req);
    }

    // ── 脱敏摘要 ────────────────────────────────────

    @Test
    public void buildSummary_containsKeywordsOnly() {
        // buildSummary 是 private 方法，通过 buildRequest 间接测试
        String content = "安全账户：您的账户异常，请立即转账至安全账户";
        MultimodalRequest req = extractor.buildRequest(content, "110", null, null, "s");
        String summary = req.sms_text_summary;
        assertNotNull("摘要不为 null", summary);
        assertFalse("摘要不应包含原文", summary.contains(content));
        assertTrue("摘要应包含命中词汇 '安全账户'", summary.contains("安全账户"));
    }

    // ── 工具方法 ────────────────────────────────────

    @Test
    public void shannonEntropy_variedStrings() {
        // 间接测试 extractDomain/extractPath 等工具方法
        List<Float> urlFeatures = extractor.buildUrlFeatures(
            Arrays.asList("https://aabbccdd.com/path/deep/nested"));
        assertNotNull("urlFeatures", urlFeatures);
        assertEquals("维度 = 6", 6, urlFeatures.size());
    }
}
