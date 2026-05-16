package com.campus.safety.model;

import java.util.List;

public class FraudCase {
    public int id;
    public String title;
    public String summary;
    public String content;
    public String category;
    public String risk_level;
    public String emoji;
    public int view_count;
    public int like_count;
    public int share_count;
    public boolean is_featured;
    public boolean is_favorited;
    public List<String> tags;
    public String published_at;
}
