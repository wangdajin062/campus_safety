package com.campus.safety.model;
import java.util.List;

public class SmsAnalyzeResult {
    public String risk_level;
    public int risk_score;
    public double confidence;
    public String rule_triggered;
    public double ml_probability;
    public List<String> features_used;
    public String explanation;
}
