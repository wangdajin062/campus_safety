package com.campus.safety.service;

import android.util.Log;
import androidx.annotation.NonNull;

import com.campus.safety.network.ApiClient;
import com.campus.safety.util.NotificationHelper;
import com.campus.safety.util.TokenManager;
import com.google.firebase.messaging.FirebaseMessagingService;
import com.google.firebase.messaging.RemoteMessage;

import java.util.HashMap;
import java.util.Map;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class FcmMessageService extends FirebaseMessagingService {

    private static final String TAG = "FcmService";

    @Override
    public void onMessageReceived(@NonNull RemoteMessage msg) {
        super.onMessageReceived(msg);
        Map<String, String> data = msg.getData();
        String type  = data.getOrDefault("type", "alert");
        String title = data.getOrDefault("title", "诈骗预警");
        String body  = data.getOrDefault("body", "");
        String phone = data.getOrDefault("phone", "");

        Log.d(TAG, "FCM received: type=" + type);
        switch (type) {
            case "call_alert":
                int score = 80;
                try { score = Integer.parseInt(data.getOrDefault("risk_score", "80")); }
                catch (NumberFormatException ignored) {}
                NotificationHelper.showCallAlert(this, phone, score, null, score >= 70);
                break;
            case "sms_alert":
                NotificationHelper.showSmsAlert(this, phone, 75, body);
                break;
            default:
                NotificationHelper.showSmsAlert(this, "系统", 50, title + " — " + body);
        }
    }

    @Override
    public void onNewToken(@NonNull String token) {
        super.onNewToken(token);
        TokenManager.saveFcmToken(this, token);
        Log.d(TAG, "FCM token refreshed (len=" + token.length() + ")");

        if (!TokenManager.isLoggedIn(this)) return;
        Map<String, String> body = new HashMap<>();
        body.put("fcm_token", token);
        body.put("platform", "android");
        ApiClient.getApi().registerDevice(body).enqueue(new Callback<com.campus.safety.model.ApiResponse<Map<String, Object>>>() {
            @Override public void onResponse(Call<com.campus.safety.model.ApiResponse<Map<String, Object>>> c,
                                              Response<com.campus.safety.model.ApiResponse<Map<String, Object>>> r) {
                Log.d(TAG, "device registered: HTTP " + r.code());
            }
            @Override public void onFailure(Call<com.campus.safety.model.ApiResponse<Map<String, Object>>> c, Throwable t) {
                Log.w(TAG, "register device failed: " + t.getMessage());
            }
        });
    }
}
