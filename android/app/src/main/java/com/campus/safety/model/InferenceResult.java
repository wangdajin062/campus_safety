package com.campus.safety.model;
import java.util.Map;

public class InferenceResult {
    public String risk_level;
    public int risk_score;
    public double confidence;
    public double latency_ms;
    public String rule_triggered;
    public double ml_probability;
    public Map<String, Integer> modalities;
}
