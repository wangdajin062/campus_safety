package com.campus.safety.receiver;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.os.Build;
import android.util.Log;

import com.campus.safety.service.CallMonitorService;
import com.campus.safety.util.TokenManager;

/** 开机启动 — 自动拉起来电监听前台服务 */
public class BootReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context ctx, Intent intent) {
        String action = intent.getAction();
        if (!Intent.ACTION_BOOT_COMPLETED.equals(action)
            && !"android.intent.action.QUICKBOOT_POWERON".equals(action)) return;

        if (!TokenManager.isLoggedIn(ctx)) return;

        Intent svc = new Intent(ctx, CallMonitorService.class);
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) ctx.startForegroundService(svc);
            else ctx.startService(svc);
            Log.i("BootReceiver", "CallMonitorService started on boot");
        } catch (Exception e) {
            Log.w("BootReceiver", "Failed to start service on boot: " + e.getMessage());
        }
    }
}
