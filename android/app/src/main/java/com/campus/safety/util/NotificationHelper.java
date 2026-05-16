package com.campus.safety.util;

import android.app.Notification;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;

import androidx.core.app.NotificationCompat;

import com.campus.safety.R;
import com.campus.safety.ml.Utils;
import com.campus.safety.ui.activity.DetectionResultActivity;

/** 统一通知构造器 */
public class NotificationHelper {

    public static final String CH_URGENT  = "campus_safety_urgent";
    public static final String CH_WARN    = "campus_safety_warn";
    public static final String CH_SERVICE = "campus_safety_service";

    /** 高危来电预警（振动+灯光） */
    public static void showCallAlert(Context ctx, String phone, int riskScore,
                                      String ruleTriggered, boolean isUrgent) {
        String ch = isUrgent ? CH_URGENT : CH_WARN;
        String title = (isUrgent ? "🚨 高危诈骗来电 " : "⚡ 可疑来电 ")
                      + Utils.maskPhone(phone);
        String body = "风险评分 " + riskScore + "/100"
                     + (ruleTriggered != null ? "\n触发规则: " + ruleTriggered : "");

        Intent openIntent = new Intent(ctx, DetectionResultActivity.class)
            .putExtra("phone", phone)
            .putExtra("risk_score", riskScore)
            .putExtra("rule_triggered", ruleTriggered)
            .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);

        int flags = PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE;
        PendingIntent pi = PendingIntent.getActivity(ctx, phone.hashCode(), openIntent, flags);

        Notification n = new NotificationCompat.Builder(ctx, ch)
            .setSmallIcon(R.drawable.ic_shield)
            .setColor(isUrgent ? 0xFFF44336 : 0xFFFF9800)
            .setContentTitle(title)
            .setContentText(body)
            .setStyle(new NotificationCompat.BigTextStyle().bigText(body))
            .setPriority(isUrgent ? NotificationCompat.PRIORITY_MAX : NotificationCompat.PRIORITY_DEFAULT)
            .setCategory(NotificationCompat.CATEGORY_CALL)
            .setAutoCancel(true)
            .setContentIntent(pi)
            .build();

        NotificationManager nm = (NotificationManager) ctx.getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(phone.hashCode(), n);
    }

    /** 短信预警 */
    public static void showSmsAlert(Context ctx, String sender, int riskScore, String summary) {
        String ch = riskScore >= 70 ? CH_URGENT : CH_WARN;
        String title = (riskScore >= 70 ? "🚨 诈骗短信 " : "⚡ 可疑短信 ") + Utils.maskPhone(sender);
        Intent i = new Intent(ctx, DetectionResultActivity.class)
            .putExtra("phone", sender)
            .putExtra("risk_score", riskScore)
            .putExtra("type", "sms")
            .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        PendingIntent pi = PendingIntent.getActivity(ctx, sender.hashCode() + 10000, i,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        Notification n = new NotificationCompat.Builder(ctx, ch)
            .setSmallIcon(R.drawable.ic_shield)
            .setColor(riskScore >= 70 ? 0xFFF44336 : 0xFFFF9800)
            .setContentTitle(title)
            .setContentText(Utils.truncate(summary, 80))
            .setStyle(new NotificationCompat.BigTextStyle().bigText(summary))
            .setPriority(riskScore >= 70 ? NotificationCompat.PRIORITY_MAX : NotificationCompat.PRIORITY_DEFAULT)
            .setCategory(NotificationCompat.CATEGORY_MESSAGE)
            .setAutoCancel(true)
            .setContentIntent(pi)
            .build();

        NotificationManager nm = (NotificationManager) ctx.getSystemService(Context.NOTIFICATION_SERVICE);
        if (nm != null) nm.notify(sender.hashCode() + 10000, n);
    }

    /** 前台服务常驻通知 */
    public static Notification buildServiceNotification(Context ctx) {
        Intent openApp = new Intent(ctx, com.campus.safety.ui.activity.MainActivity.class);
        PendingIntent pi = PendingIntent.getActivity(ctx, 0, openApp,
            PendingIntent.FLAG_UPDATE_CURRENT | PendingIntent.FLAG_IMMUTABLE);

        return new NotificationCompat.Builder(ctx, CH_SERVICE)
            .setSmallIcon(R.drawable.ic_shield)
            .setContentTitle("校园安全防护运行中")
            .setContentText("正在实时监测来电与短信诈骗风险")
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .setContentIntent(pi)
            .build();
    }
}
