package com.campus.safety.model;

public class ReportRequest {
    public String target;       // 手机号
    public String risk_type;    // fraud|harassment|spam
    public String description;  // 描述
    public String evidence_url; // 可选截图 URL
}
