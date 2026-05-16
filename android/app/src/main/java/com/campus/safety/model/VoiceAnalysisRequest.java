package com.campus.safety.model;
import java.util.List;

public class VoiceAnalysisRequest {
    public List<Float> audio_embedding;
    public String voice_text;
    public double call_duration_s;
    public String phone_number;
}
