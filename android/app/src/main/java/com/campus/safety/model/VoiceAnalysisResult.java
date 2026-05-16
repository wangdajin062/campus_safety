package com.campus.safety.model;
import java.util.Map;

public class VoiceAnalysisResult {
    public String risk_level;
    public int risk_score;
    public int voice_score;
    public int text_score;
    public Map<String, Double> acoustic_indicators;
    public double call_duration_s;
}
