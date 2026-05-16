package com.campus.safety.model;

import java.util.List;

public class HomeData {
    // 新版 API 字段
    public Stats stats;
    public List<FraudAlert> latest_alerts;
    public TodayTip today_tip;

    // 兼容旧版 API 字段
    public int blocked_today;
    public int alerted_sms;
    public int fraud_phone_count;
    public int case_count;

    public static class Stats {
        public int protection_score;
        public int blocked_calls;
        public int alerted_sms;
        public int total_reports;
        public int cases_read;
        public String protection_level;
    }

    public static class TodayTip {
        public String title;
        public String content;
        public String emoji;
    }
}
