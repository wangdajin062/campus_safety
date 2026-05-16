package com.campus.safety.service;

import android.app.Service;
import android.content.Intent;
import android.os.IBinder;
import android.telephony.PhoneStateListener;
import android.telephony.TelephonyManager;
import android.util.Log;

import androidx.annotation.Nullable;

import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.PhoneCheckResult;
import com.campus.safety.network.ApiClient;
import com.campus.safety.util.NotificationHelper;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 前台来电监听服务
 * - Android 8+ 前台服务
 * - PhoneStateListener 监听 CALL_STATE_RINGING
 * - 异步查询后端 /v1/calls/check，高风险 → 系统通知
 */
public class CallMonitorService extends Service {

    private static final String TAG = "CallMonitor";
    private static final int NOTIF_ID = 2001;

    private TelephonyManager tm;
    private PhoneStateListener listener;
    private String lastChecked = "";

    @Override
    public void onCreate() {
        super.onCreate();
        startForeground(NOTIF_ID, NotificationHelper.buildServiceNotification(this));

        tm = (TelephonyManager) getSystemService(TELEPHONY_SERVICE);
        listener = new PhoneStateListener() {
            @Override
            public void onCallStateChanged(int state, String number) {
                if (state == TelephonyManager.CALL_STATE_RINGING
                        && number != null && !number.equals(lastChecked)) {
                    lastChecked = number;
                    checkIncomingCall(number);
                }
            }
        };
        try {
            tm.listen(listener, PhoneStateListener.LISTEN_CALL_STATE);
        } catch (SecurityException e) {
            Log.w(TAG, "READ_PHONE_STATE permission denied");
        }
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) { return START_STICKY; }

    private void checkIncomingCall(String phone) {
        Log.d(TAG, "Incoming call from " + phone.substring(0, Math.min(3, phone.length())) + "****");
        ApiClient.getApi().checkPhone(phone).enqueue(new Callback<ApiResponse<PhoneCheckResult>>() {
            @Override public void onResponse(Call<ApiResponse<PhoneCheckResult>> c, Response<ApiResponse<PhoneCheckResult>> r) {
                if (!r.isSuccessful() || r.body() == null || r.body().data == null) return;
                PhoneCheckResult pr = r.body().data;
                if ("high".equals(pr.risk_level) || "medium".equals(pr.risk_level)) {
                    NotificationHelper.showCallAlert(CallMonitorService.this, phone,
                        pr.risk_score, null, "high".equals(pr.risk_level));
                }
            }
            @Override public void onFailure(Call<ApiResponse<PhoneCheckResult>> c, Throwable t) {
                Log.w(TAG, "check call failed: " + t.getMessage());
            }
        });
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (tm != null && listener != null) {
            try { tm.listen(listener, PhoneStateListener.LISTEN_NONE); } catch (Exception ignored) {}
        }
    }

    @Nullable @Override public IBinder onBind(Intent intent) { return null; }
}
