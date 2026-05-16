package com.campus.safety.ui.activity;

import android.os.Bundle;
import android.view.View;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.campus.safety.BuildConfig;
import com.campus.safety.databinding.ActivitySettingsBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.ModelStatus;
import com.campus.safety.network.ApiClient;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 设置
 * - 显示模型状态
 * - 切换通知/后台监测开关
 * - 关于页
 */
public class SettingsActivity extends AppCompatActivity {
    private ActivitySettingsBinding bd;

    @Override
    protected void onCreate(Bundle s) {
        super.onCreate(s);
        bd = ActivitySettingsBinding.inflate(getLayoutInflater());
        setContentView(bd.getRoot());
        bd.toolbar.setNavigationOnClickListener(x -> finish());

        bd.tvVersion.setText("v" + BuildConfig.VERSION_NAME);
        bd.tvApiUrl.setText(BuildConfig.API_BASE_URL);

        bd.switchCallMonitor.setChecked(true);
        bd.switchSmsMonitor.setChecked(true);
        bd.switchNotifications.setChecked(true);

        bd.tvAbout.setOnClickListener(x -> {
            new androidx.appcompat.app.AlertDialog.Builder(this)
                .setTitle("关于校园安全")
                .setMessage("校园安全 APP v3.0\n\n软硬协同多模态电信欺诈检测系统\n基于推测解码 + 量化感知蒸馏\n\n©2025 校园安全团队")
                .setPositiveButton("确定", null)
                .show();
        });

        bd.tvModelStatus.setOnClickListener(x -> loadModelStatus());
        loadModelStatus();
    }

    private void loadModelStatus() {
        ApiClient.getApi().getModelStatus().enqueue(new Callback<ApiResponse<ModelStatus>>() {
            @Override public void onResponse(Call<ApiResponse<ModelStatus>> c, Response<ApiResponse<ModelStatus>> r) {
                if (!r.isSuccessful() || r.body() == null || r.body().data == null) return;
                ModelStatus m = r.body().data;
                StringBuilder sb = new StringBuilder();
                if (m.draft_model != null) {
                    sb.append("端侧草稿模型: ").append(m.draft_model.get("type")).append("\n");
                    sb.append("大小: ").append(m.draft_model.get("size_mb")).append(" MB\n");
                }
                if (m.runtime_stats != null) {
                    sb.append("接受率: ").append(m.runtime_stats.get("acceptance_rate")).append("\n");
                    sb.append("加速比: ").append(m.runtime_stats.get("speedup_factor")).append("x\n");
                }
                bd.tvModelInfo.setText(sb.toString());
            }
            @Override public void onFailure(Call<ApiResponse<ModelStatus>> c, Throwable t) {
                bd.tvModelInfo.setText("无法获取模型状态");
            }
        });
    }
}
