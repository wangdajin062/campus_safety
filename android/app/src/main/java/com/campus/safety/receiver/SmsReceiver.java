package com.campus.safety.receiver;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.provider.Telephony;
import android.telephony.SmsMessage;
import android.util.Log;

import com.campus.safety.ml.SmsFeatureExtractor;
import com.campus.safety.ml.SmsFeatureExtractor.Features;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.SmsAnalyzeRequest;
import com.campus.safety.model.SmsAnalyzeResult;
import com.campus.safety.network.ApiClient;
import com.campus.safety.util.NotificationHelper;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 短信广播接收器 (priority=999)
 * - 拦截 SMS_RECEIVED 广播
 * - 端侧特征提取（<5ms，原文立即 GC，不上传）
 * - 本地 quickRisk 高危 → 立即通知
 * - 异步上报特征向量到 /v1/sms/analyze 获取后端精细判定
 */
public class SmsReceiver extends BroadcastReceiver {

    private static final String TAG = "SmsReceiver";

    @Override
    public void onReceive(final Context ctx, Intent intent) {
        if (!Telephony.Sms.Intents.SMS_RECEIVED_ACTION.equals(intent.getAction())) return;

        SmsMessage[] msgs;
        try {
            msgs = Telephony.Sms.Intents.getMessagesFromIntent(intent);
        } catch (Exception e) {
            Log.e(TAG, "SMS parse failed", e);
            return;
        }
        if (msgs == null || msgs.length == 0) return;

        final String sender = msgs[0].getOriginatingAddress();
        StringBuilder bodyBuilder = new StringBuilder();
        for (SmsMessage m : msgs) if (m.getMessageBody() != null) bodyBuilder.append(m.getMessageBody());
        final String body = bodyBuilder.toString();
        if (body.isEmpty()) return;

        // 端侧特征提取（原文不上传，方法返回后立即 GC）
        final Features f = SmsFeatureExtractor.extract(body, sender);
        final String quickLevel = f.quickRiskLevel();
        final int quickScore = f.quickScore();

        // 本地高危 → 立即系统通知（即使无网络也能预警）
        if ("high".equals(quickLevel)) {
            String hint = f.ruleTriggered != null
                ? "触发规则：" + f.ruleTriggered
                : "检测到多个高风险关键词";
            NotificationHelper.showSmsAlert(ctx, sender, quickScore, hint);
        }

        // 异步上传特征向量（12 维，不含原文）
        SmsAnalyzeRequest req = new SmsAnalyzeRequest();
        req.sender          = sender;
        req.keywords        = f.hitKeywords;
        req.has_url         = f.hasUrl;
        req.url_count       = f.urlCount;
        req.urgency_score   = f.urgencyScore;
        req.money_mentioned = f.moneyMentioned;
        req.impersonation   = f.impersonation;
        req.char_count      = f.charCount;

        ApiClient.getApi().analyzeSms(req).enqueue(new Callback<ApiResponse<SmsAnalyzeResult>>() {
            @Override
            public void onResponse(Call<ApiResponse<SmsAnalyzeResult>> c,
                                   Response<ApiResponse<SmsAnalyzeResult>> r) {
                if (!r.isSuccessful() || r.body() == null || r.body().data == null) return;
                SmsAnalyzeResult sr = r.body().data;
                // 服务端发现高危而本地未发现：补推通知
                if ("high".equals(sr.risk_level) && !"high".equals(quickLevel)) {
                    NotificationHelper.showSmsAlert(ctx, sender, sr.risk_score,
                        sr.explanation != null ? sr.explanation : "后端检测到诈骗风险");
                }
            }
            @Override public void onFailure(Call<ApiResponse<SmsAnalyzeResult>> c, Throwable t) {
                Log.w(TAG, "SMS analyze failed: " + t.getMessage());
            }
        });
    }
}
