package com.campus.safety.ui.activity;

import android.os.Bundle;
import android.view.View;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.campus.safety.R;
import com.campus.safety.databinding.ActivityReportBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.ReportRequest;
import com.campus.safety.network.ApiClient;
import java.util.Map;
import java.util.regex.Pattern;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class ReportActivity extends AppCompatActivity {
    private static final Pattern PHONE_RE = Pattern.compile("^1[3-9]\\d{9}$");
    private ActivityReportBinding bd;

    @Override
    protected void onCreate(Bundle s) {
        super.onCreate(s);
        bd = ActivityReportBinding.inflate(getLayoutInflater());
        setContentView(bd.getRoot());
        bd.toolbar.setNavigationOnClickListener(x -> finish());

        String presetPhone = getIntent().getStringExtra("phone");
        if (presetPhone != null) bd.etTarget.setText(presetPhone);

        bd.btnSubmit.setOnClickListener(x -> submit());
    }

    private void submit() {
        String target = bd.etTarget.getText().toString().trim();
        String desc = bd.etDescription.getText().toString().trim();
        if (!PHONE_RE.matcher(target).matches()) {
            Toast.makeText(this, "请输入正确手机号", Toast.LENGTH_SHORT).show();
            return;
        }
        if (desc.length() < 5) {
            Toast.makeText(this, "描述至少5个字", Toast.LENGTH_SHORT).show();
            return;
        }

        int typeId = bd.rgType.getCheckedRadioButtonId();
        String type = (typeId == R.id.rb_fraud) ? "fraud"
                    : (typeId == R.id.rb_harass) ? "harassment" : "spam";

        ReportRequest req = new ReportRequest();
        req.target = target;
        req.risk_type = type;
        req.description = desc;

        bd.btnSubmit.setEnabled(false);
        bd.progress.setVisibility(View.VISIBLE);
        ApiClient.getApi().submitReport(req).enqueue(new Callback<ApiResponse<Map<String, Object>>>() {
            @Override public void onResponse(Call<ApiResponse<Map<String, Object>>> c, Response<ApiResponse<Map<String, Object>>> r) {
                bd.progress.setVisibility(View.GONE);
                if (r.isSuccessful() && r.body() != null && r.body().isSuccess()) {
                    Toast.makeText(ReportActivity.this, "✅ 举报已提交，感谢您的贡献", Toast.LENGTH_LONG).show();
                    finish();
                } else {
                    bd.btnSubmit.setEnabled(true);
                    Toast.makeText(ReportActivity.this, "提交失败，请重试", Toast.LENGTH_SHORT).show();
                }
            }
            @Override public void onFailure(Call<ApiResponse<Map<String, Object>>> c, Throwable t) {
                bd.progress.setVisibility(View.GONE);
                bd.btnSubmit.setEnabled(true);
                Toast.makeText(ReportActivity.this, "网络错误", Toast.LENGTH_SHORT).show();
            }
        });
    }
}
