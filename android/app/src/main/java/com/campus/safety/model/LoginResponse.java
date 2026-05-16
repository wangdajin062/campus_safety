package com.campus.safety.model;

public class LoginResponse {
    public String token;
    public User user;

    public static class User {
        public long id;
        public String phone;
        public String nickname;
        public String avatar_url;
        public int protection_score;
        public int blocked_calls;
        public int alerted_sms;
        public int total_reports;
        public int cases_read;
        public boolean is_new_user;
    }
}
