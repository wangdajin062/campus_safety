package com.campus.safety.model;
import java.util.List;

public class SmsAnalyzeRequest {
    public String sender;
    public List<String> keywords;   // 端侧提取的命中关键词
    public boolean has_url;
    public int url_count;
    public double urgency_score;
    public boolean money_mentioned;
    public boolean impersonation;
    public int char_count;
    // 注意：原文不上传
}
