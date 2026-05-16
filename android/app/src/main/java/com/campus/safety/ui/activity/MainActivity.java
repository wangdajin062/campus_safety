package com.campus.safety.ui.activity;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.localbroadcastmanager.content.LocalBroadcastManager;

import com.campus.safety.R;
import com.campus.safety.databinding.ActivityMainBinding;
import com.campus.safety.network.interceptor.ErrorInterceptor;
import com.campus.safety.service.CallMonitorService;
import com.campus.safety.ui.fragment.AlertsFragment;
import com.campus.safety.ui.fragment.CallCheckFragment;
import com.campus.safety.ui.fragment.CasesFragment;
import com.campus.safety.ui.fragment.HomeFragment;
import com.campus.safety.ui.fragment.ProfileFragment;

import com.google.android.material.bottomnavigation.BottomNavigationView;

import java.util.ArrayList;
import java.util.List;

/**
 * 主 Activity
 * ============
 * - BottomNavigation 5 个 tab：首页 / 来电 / 案例 / 预警 / 我的
 * - 运行时权限批量请求（4 个危险权限）
 * - 登录态监听：收到 ACTION_UNAUTHORIZED 时自动跳转 Login
 * - 网络错误 Toast
 * - Fragment 切换保留状态（show/hide 模式）
 */
public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding bd;
    private final List<Fragment> fragments = new ArrayList<>(5);
    private int currentIdx = 0;

    private final androidx.activity.result.ActivityResultLauncher<String[]> permsLauncher =
        registerForActivityResult(new ActivityResultContracts.RequestMultiplePermissions(),
            result -> {
                int granted = 0;
                for (Boolean ok : result.values()) if (ok != null && ok) granted++;
                if (granted > 0) {
                    Toast.makeText(this, "已授权 " + granted + " 项权限，防护已启动", Toast.LENGTH_SHORT).show();
                    startMonitorService();
                }
            });

    private final BroadcastReceiver errorReceiver = new BroadcastReceiver() {
        @Override
        public void onReceive(Context context, Intent intent) {
            String action = intent.getAction();
            if (ErrorInterceptor.ACTION_UNAUTHORIZED.equals(action)) {
                Toast.makeText(MainActivity.this, "登录已过期，请重新登录", Toast.LENGTH_SHORT).show();
                startActivity(new Intent(MainActivity.this, LoginActivity.class)
                    .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK));
                finish();
            } else if (ErrorInterceptor.ACTION_RATE_LIMITED.equals(action)) {
                Toast.makeText(MainActivity.this, "请求过于频繁，请稍后再试", Toast.LENGTH_SHORT).show();
            } else if (ErrorInterceptor.ACTION_SERVER_ERROR.equals(action)) {
                Toast.makeText(MainActivity.this, "服务器暂时不可用", Toast.LENGTH_SHORT).show();
            }
        }
    };

    @Override
    protected void onCreate(Bundle s) {
        super.onCreate(s);
        bd = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(bd.getRoot());

        // 初始化 Fragments
        fragments.add(new HomeFragment());
        fragments.add(new CallCheckFragment());
        fragments.add(new CasesFragment());
        fragments.add(new AlertsFragment());
        fragments.add(new ProfileFragment());

        getSupportFragmentManager()
            .beginTransaction()
            .add(R.id.nav_host_fragment, fragments.get(0), "f0")
            .commit();

        bd.bottomNav.setOnItemSelectedListener(this::onNavSelected);

        // 权限请求
        requestRuntimePermissions();
    }

    @Override
    protected void onResume() {
        super.onResume();
        IntentFilter f = new IntentFilter();
        f.addAction(ErrorInterceptor.ACTION_UNAUTHORIZED);
        f.addAction(ErrorInterceptor.ACTION_RATE_LIMITED);
        f.addAction(ErrorInterceptor.ACTION_SERVER_ERROR);
        LocalBroadcastManager.getInstance(this).registerReceiver(errorReceiver, f);
    }

    @Override
    protected void onPause() {
        super.onPause();
        LocalBroadcastManager.getInstance(this).unregisterReceiver(errorReceiver);
    }

    private boolean onNavSelected(MenuItem item) {
        int id = item.getItemId();
        int idx;
        if (id == R.id.nav_home) idx = 0;
        else if (id == R.id.nav_call_check) idx = 1;
        else if (id == R.id.nav_cases) idx = 2;
        else if (id == R.id.nav_alerts) idx = 3;
        else if (id == R.id.nav_profile) idx = 4;
        else return false;

        if (idx == currentIdx) return true;

        Fragment current = fragments.get(currentIdx);
        Fragment target  = fragments.get(idx);

        androidx.fragment.app.FragmentTransaction tx =
            getSupportFragmentManager().beginTransaction();
        tx.setCustomAnimations(R.anim.fade_in, R.anim.fade_out);

        if (!target.isAdded()) tx.add(R.id.nav_host_fragment, target, "f" + idx);
        tx.hide(current).show(target).commit();

        currentIdx = idx;
        return true;
    }

    private void requestRuntimePermissions() {
        List<String> need = new ArrayList<>();
        String[] perms = {
            android.Manifest.permission.READ_PHONE_STATE,
            android.Manifest.permission.READ_CALL_LOG,
            android.Manifest.permission.RECEIVE_SMS,
            android.Manifest.permission.READ_SMS,
        };
        for (String p : perms) {
            if (ContextCompat.checkSelfPermission(this, p) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                need.add(p);
            }
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.POST_NOTIFICATIONS)
                != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                need.add(android.Manifest.permission.POST_NOTIFICATIONS);
            }
        }
        if (!need.isEmpty()) permsLauncher.launch(need.toArray(new String[0]));
        else startMonitorService();
    }

    private void startMonitorService() {
        Intent svc = new Intent(this, CallMonitorService.class);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(svc);
        } else {
            startService(svc);
        }
    }

    public void switchToTab(int idx) {
        if (idx < 0 || idx >= fragments.size()) return;
        int menuId = new int[]{R.id.nav_home, R.id.nav_call_check, R.id.nav_cases,
                                R.id.nav_alerts, R.id.nav_profile}[idx];
        bd.bottomNav.setSelectedItemId(menuId);
    }
}
