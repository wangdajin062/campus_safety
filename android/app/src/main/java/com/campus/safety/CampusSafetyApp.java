package com.campus.safety;

import android.app.Application;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.os.Build;
import android.util.Log;

import com.campus.safety.engine.OnDeviceLLMEngine;
import com.campus.safety.network.ApiClient;
import com.campus.safety.util.TokenManager;

/**
 * Application 全局入口
 * ====================
 * 负责：
 *   1. 初始化 Retrofit / OkHttp 客户端
 *   2. 初始化端侧 LLM 引擎（异步加载，不阻塞启动）
 *   3. 创建通知渠道
 *   4. 配置 Firebase Cloud Messaging
 */
public class CampusSafetyApp extends Application {

    private static final String TAG = "CampusSafetyApp";
    private static CampusSafetyApp instance;

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;

        long t0 = System.currentTimeMillis();

        // 1. 初始化 API 客户端（含 Token 自动注入 interceptor）
        ApiClient.init(this);

        // 2. 创建通知渠道（Android 8+）
        createNotificationChannels();

        // 3. 异步加载端侧 LLM 草稿模型（不阻塞应用启动）
        if (TokenManager.isLoggedIn(this)) {
            OnDeviceLLMEngine.getInstance().loadModelAsync(this, (success, size) -> {
                Log.i(TAG, "Draft model " + (success ? "loaded (" + size/1024/1024 + "MB)" : "fallback to statistical prior"));
            });
        }

        Log.i(TAG, "CampusSafety App initialized in " + (System.currentTimeMillis() - t0) + "ms");
    }

    private void createNotificationChannels() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return;

        NotificationManager nm = getSystemService(NotificationManager.class);
        if (nm == null) return;

        // 高危预警渠道（振动+声音）
        NotificationChannel urgent = new NotificationChannel(
            "campus_safety_urgent",
            "高危诈骗预警",
            NotificationManager.IMPORTANCE_HIGH
        );
        urgent.setDescription("检测到高危诈骗来电/短信时立即推送");
        urgent.enableVibration(true);
        urgent.setVibrationPattern(new long[]{0, 300, 100, 300, 100, 300});
        nm.createNotificationChannel(urgent);

        // 中危提醒渠道
        NotificationChannel warn = new NotificationChannel(
            "campus_safety_warn",
            "诈骗风险提醒",
            NotificationManager.IMPORTANCE_DEFAULT
        );
        warn.setDescription("可疑来电/短信提醒");
        warn.enableVibration(true);
        warn.setVibrationPattern(new long[]{0, 200});
        nm.createNotificationChannel(warn);

        // 服务常驻渠道（低优先级）
        NotificationChannel service = new NotificationChannel(
            "campus_safety_service",
            "后台防护服务",
            NotificationManager.IMPORTANCE_LOW
        );
        service.setDescription("后台实时防护运行中");
        service.setShowBadge(false);
        nm.createNotificationChannel(service);
    }

    public static CampusSafetyApp getInstance() { return instance; }
}
