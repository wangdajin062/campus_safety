package com.campus.safety.model;
import java.util.List;

public class FeedbackRequest {
    public String sample_hash;
    public int true_label;   // 0=safe, 1=fraud
    public List<Float> feature_vector;
    public String text_summary;
}
