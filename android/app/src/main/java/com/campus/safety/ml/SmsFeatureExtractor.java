package com.campus.safety.ml;

import android.util.Log;

import com.campus.safety.model.MultimodalRequest;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * SmsFeatureExtractor — QAD-MultiGuard v4.1
 * ===========================================
 * 端侧 SMS / 来电特征提取器
 *
 * 升级（v4 → v4.1）：
 *   ✓ buildSmsFeatures()  → 12 维向量（与后端 SmsFeatures.vectorize_sms 对齐）
 *   ✓ buildUrlFeatures()  → 6 维向量（新增，支持 URL 模态）
 *   ✓ buildCallFeatures() → 12 维向量（与后端 PhoneFeatures.vectorize_phone 对齐）
 *   ✓ buildRequest()      → 一步构建完整 MultimodalRequest
 *
 * 隐私保证：
 *   提取特征向量后，短信原文不应上传，
 *   只允许上传脱敏摘要（sms_text_summary）。
 */
public class SmsFeatureExtractor {

    private static final String TAG = "SmsFeatureExtractor";

    // ── 高权重欺诈词汇（对应后端 RuleEngine.KEYWORD_WEIGHT_MAP）────
    private static final String[] HIGH_KEYWORDS = {
        "安全账户", "立即转账", "公安局", "涉案资金", "资产冻结",
        "配合调查", "刷单", "刷好评", "解冻", "助学贷款",
        "内部名额", "恭喜中奖", "账户异常", "点击链接"
    };
    private static final int[] HIGH_WEIGHTS = {
        100, 90, 90, 92, 88,
        85,  85, 80, 80, 70,
        65,  80, 75, 70
    };

    private static final String[] MED_KEYWORDS = {
        "验证码", "贷款", "兼职", "转账", "账户", "密码"
    };

    // URL 正则
    private static final Pattern URL_PATTERN =
        Pattern.compile("(https?://|www\\.|bit\\.ly|t\\.cn)[^\\s]+",
                        Pattern.CASE_INSENSITIVE);

    // 短链接服务
    private static final List<String> SHORTENED_SERVICES = Arrays.asList(
        "bit.ly", "tinyurl.com", "t.cn", "goo.gl", "ow.ly", "is.gd"
    );

    // ── 短信特征提取（12 维）─────────────────────────────
    /**
     * 从短信内容提取 12 维特征向量
     *
     * 向量格式（对应后端 vectorize_sms）：
     *   [0] keyword_hits / 5.0（截止 1.0）
     *   [1] keyword_weight / 100.0
     *   [2] urgency_score
     *   [3] has_url (0/1)
     *   [4] url_count / 3.0
     *   [5] money_mentioned (0/1)
     *   [6] impersonation (0/1)
     *   [7] char_count / 300.0
     *   [8] digit_ratio
     *   [9] sender_is_number (0/1)
     *  [10] keyword_hits>0 AND has_url (0/1)
     *  [11] impersonation AND money_mentioned (0/1)
     */
    public List<Float> buildSmsFeatures(String content, String sender) {
        if (content == null) content = "";
        if (sender  == null) sender  = "";

        // 关键词命中
        int   hitCount = 0;
        float totalWeight = 0;
        for (int i = 0; i < HIGH_KEYWORDS.length; i++) {
            if (content.contains(HIGH_KEYWORDS[i])) {
                hitCount++;
                totalWeight += HIGH_WEIGHTS[i];
            }
        }
        for (String kw : MED_KEYWORDS) {
            if (content.contains(kw)) {
                hitCount++;
                totalWeight += 40;
            }
        }

        // URL
        Matcher m      = URL_PATTERN.matcher(content);
        int urlCount   = 0;
        while (m.find()) urlCount++;
        boolean hasUrl = urlCount > 0;

        // 紧迫感（感叹号、紧急词汇）
        float urgency  = 0.0f;
        String[] urgWords = {"立即", "马上", "现在", "紧急", "警告", "限时", "最后"};
        for (String w : urgWords) if (content.contains(w)) urgency += 0.15f;
        urgency += Math.min(0.3f, countChar(content, '！') * 0.1f);
        urgency  = Math.min(1.0f, urgency);

        // 涉钱
        boolean hasMoney = content.contains("万元") || content.contains("元")
                        || content.contains("¥") || content.contains("￥")
                        || content.contains("转账") || content.contains("汇款");

        // 冒充
        boolean impersonation = content.contains("公安") || content.contains("警察")
                             || content.contains("银行") || content.contains("客服")
                             || content.contains("检察") || content.contains("法院");

        // 发件人是否为号码
        boolean senderIsNum = sender.replaceAll("[+\\s-]", "").matches("\\d{7,}");

        // 数字比例
        long digits = content.chars().filter(Character::isDigit).count();
        float digitRatio = content.length() > 0
            ? (float) digits / content.length() : 0.0f;

        float kw  = Math.min(1.0f, hitCount / 5.0f);
        float kww = Math.min(1.0f, totalWeight / 100.0f);

        return toFloatList(new float[]{
            kw, kww, urgency,
            hasUrl ? 1.0f : 0.0f,
            Math.min(1.0f, urlCount / 3.0f),
            hasMoney ? 1.0f : 0.0f,
            impersonation ? 1.0f : 0.0f,
            Math.min(1.0f, content.length() / 300.0f),
            digitRatio,
            senderIsNum ? 1.0f : 0.0f,
            (hitCount > 0 && hasUrl) ? 1.0f : 0.0f,
            (impersonation && hasMoney) ? 1.0f : 0.0f,
        });
    }

    // ── URL 特征提取（6 维）──────────────────────────────
    /**
     * 从 URL 列表提取 6 维特征向量（对应后端 url_features 格式）
     *
     * 向量格式：
     *   [0] avg_domain_len / 100.0
     *   [1] avg_path_depth / 10.0
     *   [2] any_url_has_ip_as_domain (0/1)
     *   [3] any_url_has_non_std_port (0/1)
     *   [4] avg_entropy / 5.0
     *   [5] any_url_is_shortened (0/1)
     */
    public List<Float> buildUrlFeatures(List<String> urls) {
        if (urls == null || urls.isEmpty()) return null;

        float totalDomainLen = 0;
        float totalDepth     = 0;
        boolean hasIp        = false;
        boolean hasPort      = false;
        boolean isShortened  = false;
        float   totalEntropy = 0;

        int n = Math.min(urls.size(), 5);
        for (int i = 0; i < n; i++) {
            String url = urls.get(i).toLowerCase();

            // 域名长度
            String domain = extractDomain(url);
            totalDomainLen += domain.length();

            // 路径深度
            String path   = extractPath(url);
            int depth     = path.isEmpty() ? 0 : path.split("/").length - 1;
            totalDepth   += depth;

            // IP 地址
            if (domain.matches("\\d{1,3}(\\.\\d{1,3}){3}")) hasIp = true;

            // 非标端口
            if (url.matches(".*:\\d{4,5}/.*") &&
                !url.startsWith("http://") && !url.startsWith("https://"))
                hasPort = true;

            // 短链接
            for (String s : SHORTENED_SERVICES)
                if (url.contains(s)) { isShortened = true; break; }

            // 域名信息熵
            totalEntropy += shannonEntropy(domain);
        }

        return toFloatList(new float[]{
            Math.min(1.0f, (totalDomainLen / n) / 100.0f),
            Math.min(1.0f, (totalDepth / n)     / 10.0f),
            hasIp       ? 1.0f : 0.0f,
            hasPort     ? 1.0f : 0.0f,
            Math.min(1.0f, (totalEntropy / n)   / 5.0f),
            isShortened ? 1.0f : 0.0f,
        });
    }

    // ── 来电特征（12 维）────────────────────────────────
    /**
     * 从后端返回的号码风险数据构建 12 维特征（对应 vectorize_phone）
     *
     * @param reportCount    举报次数
     * @param confirmedCount 确认诈骗次数
     * @param queryCount     查询次数
     * @param daysSinceFirst 首次举报距今天数
     * @param sourceScore    数据源可信度 (user=0.3, system=0.6, police=1.0)
     */
    public List<Float> buildCallFeatures(
        int    reportCount,
        int    confirmedCount,
        int    queryCount,
        float  daysSinceFirst,
        float  sourceScore
    ) {
        boolean highReport = reportCount >= 10;
        boolean confirmed  = confirmedCount >= 3;
        boolean reportNoConfirm = reportCount > 0 && confirmedCount == 0;

        return toFloatList(new float[]{
            Math.min(1.0f, reportCount    / 50.0f),
            Math.min(1.0f, confirmedCount / 20.0f),
            Math.min(1.0f, queryCount     / 100.0f),
            Math.min(1.0f, daysSinceFirst / 365.0f),
            sourceScore,
            0.0f,   // location_code（暂不使用）
            0.0f,   // carrier_code
            highReport     ? 1.0f : 0.0f,
            confirmed      ? 1.0f : 0.0f,
            reportNoConfirm? 1.0f : 0.0f,
            (float) (Math.log1p(reportCount) / 6.0),
            sourceScore >= 1.0f ? 1.0f : 0.0f,  // police_flag
        });
    }

    // ── 一步构建完整请求 ──────────────────────────────────
    /**
     * 从短信内容一步构建完整 MultimodalRequest
     *
     * @param content        短信内容（原文不上传，仅用于本地提取特征）
     * @param sender         发件方（号码或名称）
     * @param urls           内容中提取的 URL 列表
     * @param audioEmbedding 声学嵌入 F_v（128-d，OnDeviceLLMEngine 提取）
     * @param sessionId      会话 ID
     */
    public MultimodalRequest buildRequest(
        String       content,
        String       sender,
        List<String> urls,
        List<Float>  audioEmbedding,
        String       sessionId
    ) {
        List<Float> smsFeats = buildSmsFeatures(content, sender);
        List<Float> urlFeats = (urls != null && !urls.isEmpty())
                               ? buildUrlFeatures(urls) : null;

        // 脱敏摘要：只保留关键词命中信息，不含原文
        String summary = buildSummary(content);

        return MultimodalRequest.builder()
            .smsFeatures(smsFeats)
            .smsSummary(summary)
            .urlFeatures(urlFeats)
            .audioEmbedding(audioEmbedding)
            .sessionId(sessionId != null ? sessionId : "")
            .enableCot(true)
            .build();
    }

    /**
     * 短信脱敏摘要（移除个人信息，保留欺诈信号词汇）
     */
    private String buildSummary(String content) {
        if (content == null || content.length() < 5) return "";
        StringBuilder sb = new StringBuilder("[摘要] 命中词汇：");
        for (String kw : HIGH_KEYWORDS) {
            if (content.contains(kw)) sb.append(kw).append(" ");
        }
        for (String kw : MED_KEYWORDS) {
            if (content.contains(kw)) sb.append(kw).append(" ");
        }
        // 截断，不超过 100 字
        String result = sb.toString().trim();
        return result.length() > 100 ? result.substring(0, 100) : result;
    }

    // ── 工具方法 ──────────────────────────────────────────
    private static List<Float> toFloatList(float[] arr) {
        List<Float> list = new ArrayList<>(arr.length);
        for (float v : arr) list.add(v);
        return list;
    }

    private static int countChar(String s, char c) {
        int n = 0;
        for (char x : s.toCharArray()) if (x == c) n++;
        return n;
    }

    private static String extractDomain(String url) {
        try {
            String s = url.replaceFirst("https?://", "");
            int slash = s.indexOf('/');
            return slash > 0 ? s.substring(0, slash) : s;
        } catch (Exception e) { return url; }
    }

    private static String extractPath(String url) {
        try {
            String s = url.replaceFirst("https?://[^/]+", "");
            int q = s.indexOf('?');
            return q > 0 ? s.substring(0, q) : s;
        } catch (Exception e) { return ""; }
    }

    private static float shannonEntropy(String s) {
        if (s == null || s.isEmpty()) return 0f;
        int[] freq = new int[128];
        for (char c : s.toCharArray()) if (c < 128) freq[c]++;
        float h = 0;
        int n = s.length();
        for (int f : freq) {
            if (f > 0) {
                float p = (float) f / n;
                h -= p * (float)(Math.log(p) / Math.log(2));
            }
        }
        return h;
    }
}
