package com.campus.safety.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import androidx.appcompat.app.AppCompatActivity;
import com.campus.safety.R;
import com.campus.safety.util.TokenManager;

/**
 * 启动页
 * - 显示品牌启动屏 1.2s
 * - 路由决策：已登录→Main，未登录→Login
 */
public class SplashActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);

        new Handler(Looper.getMainLooper()).postDelayed(() -> {
            Class<?> target = TokenManager.isLoggedIn(this)
                ? MainActivity.class : LoginActivity.class;
            startActivity(new Intent(this, target));
            overridePendingTransition(R.anim.fade_in, R.anim.fade_out);
            finish();
        }, 1200);
    }
}
