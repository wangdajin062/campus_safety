package com.campus.safety.util;

import android.content.Context;
import android.content.SharedPreferences;
import android.util.Log;

import androidx.security.crypto.EncryptedSharedPreferences;
import androidx.security.crypto.MasterKey;

/**
 * 安全 Token 存储
 * - 优先使用 EncryptedSharedPreferences (AES-256-GCM, Android Keystore 硬件支持)
 * - 失败回退为标准 SharedPreferences (MODE_PRIVATE)
 */
public class TokenManager {
    private static final String TAG = "TokenManager";
    private static final String SECURE_PREFS = "campus_safety_secure";
    private static final String FALLBACK_PREFS = "campus_safety_fb";

    private static final String KEY_TOKEN       = "auth_token";
    private static final String KEY_USER_ID     = "user_id";
    private static final String KEY_NICKNAME    = "nickname";
    private static final String KEY_PHONE       = "phone";
    private static final String KEY_FCM_TOKEN   = "fcm_token";
    private static final String KEY_PROT_SCORE  = "protection_score";

    private static SharedPreferences getPrefs(Context ctx) {
        try {
            MasterKey key = new MasterKey.Builder(ctx)
                .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
                .build();
            return EncryptedSharedPreferences.create(
                ctx, SECURE_PREFS, key,
                EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
                EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM);
        } catch (Exception e) {
            Log.w(TAG, "Keystore unavailable, falling back: " + e.getMessage());
            return ctx.getSharedPreferences(FALLBACK_PREFS, Context.MODE_PRIVATE);
        }
    }

    public static void saveLogin(Context ctx, String token, long uid, String nick, String phone) {
        getPrefs(ctx).edit()
            .putString(KEY_TOKEN, token)
            .putLong(KEY_USER_ID, uid)
            .putString(KEY_NICKNAME, nick == null ? "" : nick)
            .putString(KEY_PHONE, phone == null ? "" : phone)
            .apply();
    }

    public static String getToken(Context ctx) { return getPrefs(ctx).getString(KEY_TOKEN, null); }
    public static long   getUserId(Context ctx) { return getPrefs(ctx).getLong(KEY_USER_ID, -1); }
    public static String getNickname(Context ctx) { return getPrefs(ctx).getString(KEY_NICKNAME, "用户"); }
    public static String getPhone(Context ctx) { return getPrefs(ctx).getString(KEY_PHONE, ""); }

    public static boolean isLoggedIn(Context ctx) {
        String t = getToken(ctx);
        return t != null && !t.isEmpty();
    }

    public static void saveFcmToken(Context ctx, String fcmToken) {
        getPrefs(ctx).edit().putString(KEY_FCM_TOKEN, fcmToken).apply();
    }
    public static String getFcmToken(Context ctx) { return getPrefs(ctx).getString(KEY_FCM_TOKEN, null); }

    public static void saveProtectionScore(Context ctx, int score) {
        getPrefs(ctx).edit().putInt(KEY_PROT_SCORE, score).apply();
    }
    public static int getProtectionScore(Context ctx) {
        return getPrefs(ctx).getInt(KEY_PROT_SCORE, 60);
    }

    public static void clear(Context ctx) { getPrefs(ctx).edit().clear().apply(); }
}
