package com.campus.safety.ui.activity;

import android.content.Intent;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.campus.safety.R;
import com.campus.safety.databinding.ActivityLoginBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.LoginResponse;
import com.campus.safety.network.ApiClient;
import com.campus.safety.util.TokenManager;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 登录 / 注册
 * ============
 * 流程：
 *   1. 用户输入手机号（正则校验）
 *   2. 点击"获取验证码" → POST /v1/auth/send-code
 *      - 60s 倒计时 + 防抖
 *      - 429 Rate Limit → Toast 提示
 *   3. 输入 6 位验证码 → 按钮激活
 *   4. 点击"登录" → POST /v1/auth/login
 *      - 成功：保存 token + 跳转 MainActivity
 *      - 失败：验证码错误提示
 */
public class LoginActivity extends AppCompatActivity {

    private static final Pattern PHONE_RE = Pattern.compile("^1[3-9]\\d{9}$");
    private ActivityLoginBinding bd;
    private CountDownTimer countdown;

    @Override
    protected void onCreate(Bundle s) {
        super.onCreate(s);
        bd = ActivityLoginBinding.inflate(getLayoutInflater());
        setContentView(bd.getRoot());

        bd.etPhone.addTextChangedListener(new SimpleWatcher(this::updateButtons));
        bd.etCode.addTextChangedListener(new SimpleWatcher(this::updateButtons));

        bd.btnSendCode.setOnClickListener(v -> sendCode());
        bd.btnLogin.setOnClickListener(v -> login());
        bd.cbAgree.setOnCheckedChangeListener((x, c) -> updateButtons());
        bd.tvUserAgreement.setOnClickListener(v -> showAgreement("用户协议"));
        bd.tvPrivacyPolicy.setOnClickListener(v -> showAgreement("隐私政策"));

        updateButtons();
    }

    private void updateButtons() {
        String phone = bd.etPhone.getText().toString();
        String code  = bd.etCode.getText().toString();
        boolean phoneValid = PHONE_RE.matcher(phone).matches();
        boolean codeValid  = code.length() == 6;
        boolean agreed     = bd.cbAgree.isChecked();

        bd.btnSendCode.setEnabled(phoneValid && countdown == null);
        bd.btnLogin.setEnabled(phoneValid && codeValid && agreed);
    }

    private void sendCode() {
        String phone = bd.etPhone.getText().toString().trim();

        Map<String, String> body = new HashMap<>();
        body.put("phone", phone);
        bd.btnSendCode.setEnabled(false);
        ApiClient.getApi().sendCode(body).enqueue(new Callback<ApiResponse<Map<String, Object>>>() {
            @Override
            public void onResponse(Call<ApiResponse<Map<String, Object>>> c,
                                   Response<ApiResponse<Map<String, Object>>> r) {
                if (r.isSuccessful() && r.body() != null && r.body().isSuccess()) {
                    toast("验证码已发送");
                    startCountdown();
                    bd.etCode.requestFocus();
                } else if (r.code() == 429) {
                    toast("操作过于频繁，请稍后再试");
                    bd.btnSendCode.setEnabled(true);
                } else {
                    toast("发送失败，请重试");
                    bd.btnSendCode.setEnabled(true);
                }
            }

            @Override
            public void onFailure(Call<ApiResponse<Map<String, Object>>> c, Throwable t) {
                toast("网络异常：" + t.getMessage());
                bd.btnSendCode.setEnabled(true);
            }
        });
    }

    private void startCountdown() {
        countdown = new CountDownTimer(60_000, 1000) {
            @Override public void onTick(long ms) {
                bd.btnSendCode.setText((ms / 1000) + "s");
            }
            @Override public void onFinish() {
                bd.btnSendCode.setText("获取验证码");
                countdown = null;
                updateButtons();
            }
        }.start();
    }

    private void login() {
        String phone = bd.etPhone.getText().toString().trim();
        String code  = bd.etCode.getText().toString().trim();

        bd.btnLogin.setEnabled(false);
        bd.progress.setVisibility(View.VISIBLE);

        Map<String, String> body = new HashMap<>();
        body.put("phone", phone);
        body.put("code", code);

        ApiClient.getApi().login(body).enqueue(new Callback<ApiResponse<LoginResponse>>() {
            @Override
            public void onResponse(Call<ApiResponse<LoginResponse>> c, Response<ApiResponse<LoginResponse>> r) {
                bd.progress.setVisibility(View.GONE);
                if (r.isSuccessful() && r.body() != null && r.body().isSuccess()) {
                    LoginResponse lr = r.body().data;
                    TokenManager.saveLogin(LoginActivity.this, lr.token, lr.user.id,
                        lr.user.nickname, lr.user.phone);
                    TokenManager.saveProtectionScore(LoginActivity.this, lr.user.protection_score);
                    toast(lr.user.is_new_user ? "欢迎加入校园安全 🛡" : "登录成功");
                    startActivity(new Intent(LoginActivity.this, MainActivity.class)
                        .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK));
                    overridePendingTransition(R.anim.fade_in, R.anim.fade_out);
                    finish();
                } else {
                    String m = r.body() != null ? r.body().message : "验证码错误";
                    toast(m);
                    bd.etCode.setText("");
                    bd.btnLogin.setEnabled(true);
                }
            }

            @Override
            public void onFailure(Call<ApiResponse<LoginResponse>> c, Throwable t) {
                bd.progress.setVisibility(View.GONE);
                toast("登录失败：" + t.getMessage());
                bd.btnLogin.setEnabled(true);
            }
        });
    }

    private void showAgreement(String title) {
        new androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle(title)
            .setMessage((CharSequence)("用户协议".equals(title) ? getString(R.string.user_agreement) : getString(R.string.privacy_policy)))
            .setPositiveButton("我已阅读", null)
            .show();
    }

    private void toast(String msg) { Toast.makeText(this, msg, Toast.LENGTH_SHORT).show(); }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (countdown != null) { countdown.cancel(); countdown = null; }
    }

    /** 简化 TextWatcher */
    static class SimpleWatcher implements TextWatcher {
        private final Runnable onChange;
        SimpleWatcher(Runnable r) { onChange = r; }
        @Override public void beforeTextChanged(CharSequence s, int a, int b, int c) {}
        @Override public void onTextChanged(CharSequence s, int a, int b, int c) {}
        @Override public void afterTextChanged(Editable s) { onChange.run(); }
    }
}
