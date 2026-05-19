package com.campus.safety.ml;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * OnDeviceLLMEngine.quickRisk() 端侧快速风险评分单元测试
 * 覆盖：关键词评分、URL 权重、金额提及、发件人冒充、风险等级判定
 */
public class OnDeviceLLMEngineTest {

    private final OnDeviceLLMEngine engine = OnDeviceLLMEngine.getInstance();
    // quickRisk 和 quickRiskLevel 是实例方法，引擎不需要加载模型

    @Test
    public void quickRisk_emptyContent_returnsZero() {
        assertEquals("空内容 → 0", 0, engine.quickRisk(null, "10086"));
        assertEquals("空字符串 → 0", 0, engine.quickRisk("", "10086"));
    }

    @Test
    public void quickRisk_publicSecurityKeywords() {
        // 公安冒充类最高权重 95
        int score = engine.quickRisk("安全账户：您的资金已被冻结，请配合调查", "110");
        assertTrue("公安冒充关键词 → score >= 90", score >= 90);
    }

    @Test
    public void quickRisk_investmentKeywords() {
        int score = engine.quickRisk("内部消息：稳定收益8%，保本保息", "1069");
        assertTrue("投资关键词 → score >= 50", score >= 50);
    }

    @Test
    public void quickRisk_partTimeJobKeywords() {
        int score = engine.quickRisk("急招刷单兼职，每单佣金50元", "10086");
        assertTrue("刷单关键词 → score >= 50", score >= 50);
    }

    @Test
    public void quickRisk_urlBonus() {
        int noUrl = engine.quickRisk("您的账户异常", "10086");
        int withUrl = engine.quickRisk("您的账户异常，请点击 http://t.cn/abcd 查看", "10086");
        assertTrue("含 URL 应增加 35 分", withUrl >= noUrl + 30);
    }

    @Test
    public void quickRisk_moneyMention() {
        int noMoney = engine.quickRisk("您的账户异常", "10086");
        int withMoney = engine.quickRisk("您的账户异常，涉及50万元资金", "10086");
        assertTrue("涉钱应额外加分", withMoney > noMoney);
    }

    @Test
    public void quickRisk_senderImpersonation() {
        int normal = engine.quickRisk("您好，您的账单已出", "10086");
        int bankSender = engine.quickRisk("您好，您的账单已出", "中国银行");
        assertTrue("冒充银行发件人应加分", bankSender > normal);
    }

    @Test
    public void quickRisk_cappedAt100() {
        // 大量关键词叠加不应超过 100
        int score = engine.quickRisk(
            "安全账户安全账户安全账户安全账户安全账户安全账户" +
            "公安局公安局公安局立即转账立即转账解冻解冻", "110");
        assertEquals("风险评分上限 100", 100, Math.min(100, score));
    }

    @Test
    public void quickRiskLevel_high() {
        assertEquals("score 90 → high", "high", engine.quickRiskLevel(90, null));
        assertEquals("score 70 → high", "high", engine.quickRiskLevel(70, null));
        assertEquals("hardRule 非空 → high", "high", engine.quickRiskLevel(0, "规则触发"));
    }

    @Test
    public void quickRiskLevel_medium() {
        assertEquals("score 50 → medium", "medium", engine.quickRiskLevel(50, null));
        assertEquals("score 35 → medium", "medium", engine.quickRiskLevel(35, null));
    }

    @Test
    public void quickRiskLevel_safe() {
        assertEquals("score 0 → safe", "safe", engine.quickRiskLevel(0, null));
        assertEquals("score 34 → safe", "safe", engine.quickRiskLevel(34, null));
    }

    @Test
    public void quickRisk_hardRuleOverridesScore() {
        // hardRule 非空时，即使 score=0 也返回 "high"
        assertEquals("hardRule 优先", "high", engine.quickRiskLevel(0, "强制规则"));
    }

    @Test
    public void quickRisk_shortenedUrl() {
        int score = engine.quickRisk("点击链接 bit.ly/abc123 领奖", "1069");
        assertTrue("短链接应加分", score > 0);
    }

    @Test
    public void quickRisk_digitAndMoneyPattern() {
        int score = engine.quickRisk("恭喜中奖！您已获得¥50000 奖金", "1069");
        assertTrue("中奖+金额 → 高评分", score >= 50);
    }

    // ── 模型信息常量 ──────────────────────────────────

    @Test
    public void modelConstants_matchSpec() {
        assertEquals("骨干网络", "Qwen2.5-0.5B-Instruct", OnDeviceLLMEngine.BACKBONE);
        assertEquals("模型大小(MB)", 240, OnDeviceLLMEngine.MODEL_SIZE_MB);
        assertEquals("层数", 24, OnDeviceLLMEngine.N_LAYERS);
        assertEquals("隐藏维度", 896, OnDeviceLLMEngine.HIDDEN_DIM);
        assertEquals("参数量(M)", 494, OnDeviceLLMEngine.PARAMS_M);
        assertEquals("量化方案", "Q4_K_M", OnDeviceLLMEngine.QUANT_SCHEME);
        assertEquals("OV-Freeze PPL", 8.62f, OnDeviceLLMEngine.OV_FREEZE_PPL, 0.01);
        assertEquals("推理速度(tok/s)", 21.4f, OnDeviceLLMEngine.TOKENS_PER_SEC, 0.1);
    }

    @Test
    public void modelType_defaultIsStatisticalPrior() {
        assertEquals("默认 = Statistical Prior", "Statistical Prior", engine.getModelType());
    }

    @Test
    public void modelInfo_fallbackContainsAlpha() {
        String info = engine.getModelInfo();
        assertTrue("降级信息应包含 α=0.86", info.contains("0.86"));
    }
}
