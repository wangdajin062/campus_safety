package com.campus.safety.ui.activity;

import android.animation.ValueAnimator;
import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import com.campus.safety.R;
import com.campus.safety.databinding.ActivityDetectionResultBinding;
import com.campus.safety.model.CoTStreamEvent;
import com.campus.safety.model.MultimodalRequest;
import com.campus.safety.network.StreamingInferenceClient;
import com.campus.safety.ml.Utils;

import java.util.ArrayList;
import java.util.List;

/**
 * 检测结果 Activity（SSE 流式展示）
 * ==================================
 * 展示从快速结果到 CoT 推理链的完整流程：
 *   1. 第一屏：大圆环风险分数（带进度动画）
 *   2. Fast 检测卡：规则引擎 + GBM 结果（<40ms）
 *   3. 推测解码统计：接受率/加速比（<50ms）
 *   4. CoT 推理链：流式 token 追加（<300ms）
 *   5. 最终结论：红/橙/绿按钮行动引导
 *   6. 举报 + 反馈入口
 */
public class DetectionResultActivity extends AppCompatActivity {

    private ActivityDetectionResultBinding bd;
    private StreamingInferenceClient streamClient;
    private String phoneNumber;
    private int initialScore;
    private String sessionId;

    @Override
    protected void onCreate(Bundle s) {
        super.onCreate(s);
        bd = ActivityDetectionResultBinding.inflate(getLayoutInflater());
        setContentView(bd.getRoot());

        phoneNumber  = getIntent().getStringExtra("phone");
        initialScore = getIntent().getIntExtra("risk_score", 0);
        String initialLevel = getIntent().getStringExtra("risk_level");
        String ruleTriggered = getIntent().getStringExtra("rule_triggered");
        sessionId = "sess_" + System.currentTimeMillis();

        bd.tvPhone.setText(Utils.maskPhone(phoneNumber));
        bd.toolbar.setNavigationOnClickListener(x -> finish());

        // 初始 UI
        animateScore(0, initialScore);
        renderRiskLevel(initialLevel != null ? initialLevel : scoreToLevel(initialScore));
        if (ruleTriggered != null) {
            bd.tvRuleBadge.setVisibility(View.VISIBLE);
            bd.tvRuleBadge.setText("⚠ 规则触发: " + ruleTriggered);
        }

        bd.btnReport.setOnClickListener(x -> {
            startActivity(new Intent(this, ReportActivity.class).putExtra("phone", phoneNumber));
        });

        bd.btnFeedback.setOnClickListener(x -> showFeedbackDialog());
        bd.btnCancel.setOnClickListener(x -> {
            if (streamClient != null) streamClient.cancel();
            finish();
        });

        // 启动 SSE 流式检测
        startStreamDetection();
    }

    private void startStreamDetection() {
        MultimodalRequest req = new MultimodalRequest();
        req.phone_number = phoneNumber;
        req.session_id = sessionId;
        req.enable_cot = true;
        // 如需 SMS 特征，由调用方通过 Intent 传入

        streamClient = new StreamingInferenceClient(this);
        streamClient.startStream(req, new StreamingInferenceClient.StreamCallback() {
            @Override
            public void onFastResult(CoTStreamEvent e) {
                runOnUiThread(() -> {
                    bd.cardFast.setVisibility(View.VISIBLE);
                    bd.tvFastLatency.setText("⚡ " + (e.latency_ms != null ? String.format("%.0f", e.latency_ms) : "?") + " ms");
                    if (e.risk_level != null) renderRiskLevel(e.risk_level);
                    if (e.risk_score != null) animateScore(initialScore, e.risk_score);
                    if (e.modalities != null) {
                        StringBuilder sb = new StringBuilder("多模态: ");
                        e.modalities.forEach((k, v) -> sb.append(k).append("=").append(v).append(" "));
                        bd.tvModalities.setText(sb.toString().trim());
                    }
                });
            }

            @Override
            public void onImmediateAlert(CoTStreamEvent e) {
                runOnUiThread(() -> {
                    bd.cardUrgent.setVisibility(View.VISIBLE);
                    bd.tvUrgentMsg.setText("🚨 高危警告！立即挂断，切勿转账！");
                });
            }

            @Override
            public void onSpecDraft(CoTStreamEvent e) {
                runOnUiThread(() -> {
                    bd.cardSpecDraft.setVisibility(View.VISIBLE);
                    if (e.acceptance_rate != null)
                        bd.tvAcceptance.setText(String.format("接受率 %.1f%%", e.acceptance_rate * 100));
                    if (e.speedup_factor != null)
                        bd.tvSpeedup.setText(String.format("加速比 %.1fx", e.speedup_factor));
                });
            }

            @Override
            public void onCotChunk(String chunk) {
                runOnUiThread(() -> {
                    bd.cardCot.setVisibility(View.VISIBLE);
                    bd.tvCotContent.append(chunk);
                });
            }

            @Override
            public void onFinalResult(CoTStreamEvent e) {
                runOnUiThread(() -> {
                    bd.progressStream.setVisibility(View.GONE);
                    bd.btnCancel.setVisibility(View.GONE);
                    if (e.risk_score != null) animateScore(0, e.risk_score);
                    if (e.risk_level != null) renderRiskLevel(e.risk_level);
                    if (e.confidence != null)
                        bd.tvConfidence.setText(String.format("综合置信度 %.1f%%", e.confidence * 100));
                    bd.cardAction.setVisibility(View.VISIBLE);
                    String lvl = e.risk_level;
                    if ("high".equals(lvl)) {
                        bd.btnAction.setText("⚠️ 立即挂断并举报");
                        bd.btnAction.setBackgroundTintList(getResources().getColorStateList(R.color.risk_red));
                        bd.tvActionHint.setText("此号码存在高危诈骗风险，请勿回拨，切勿透露验证码、银行卡信息。");
                    } else if ("medium".equals(lvl)) {
                        bd.btnAction.setText("⚡ 提高警惕");
                        bd.btnAction.setBackgroundTintList(getResources().getColorStateList(R.color.risk_orange));
                        bd.tvActionHint.setText("保持警惕，核实对方真实身份后再做决定。");
                    } else {
                        bd.btnAction.setText("✓ 暂无风险");
                        bd.btnAction.setBackgroundTintList(getResources().getColorStateList(R.color.risk_green));
                        bd.tvActionHint.setText("暂未发现异常，仍建议谨慎应对陌生来电。");
                    }
                    bd.btnAction.setOnClickListener(v -> {
                        if ("high".equals(lvl) || "medium".equals(lvl))
                            startActivity(new Intent(DetectionResultActivity.this, ReportActivity.class)
                                .putExtra("phone", phoneNumber));
                        else finish();
                    });
                });
            }

            @Override
            public void onStreamError(String err) {
                runOnUiThread(() -> {
                    bd.progressStream.setVisibility(View.GONE);
                    Toast.makeText(DetectionResultActivity.this, "分析失败: " + err, Toast.LENGTH_LONG).show();
                });
            }

            @Override
            public void onStreamComplete() {
                runOnUiThread(() -> bd.progressStream.setVisibility(View.GONE));
            }
        });
    }

    private void animateScore(int from, int to) {
        ValueAnimator anim = ValueAnimator.ofInt(from, to);
        anim.setDuration(800);
        anim.addUpdateListener(a -> {
            int v = (int) a.getAnimatedValue();
            bd.tvScore.setText(String.valueOf(v));
            bd.progressScore.setProgress(v);
        });
        anim.start();
    }

    private void renderRiskLevel(String level) {
        int color, bg;
        String text;
        if ("high".equals(level)) { color = 0xFFF44336; bg = R.drawable.bg_risk_red;    text = "高危"; }
        else if ("medium".equals(level)) { color = 0xFFFF9800; bg = R.drawable.bg_risk_orange; text = "中危"; }
        else                              { color = 0xFF4CAF50; bg = R.drawable.bg_risk_green;  text = "安全"; }
        bd.tvRiskLevel.setText(text);
        bd.tvRiskLevel.setTextColor(color);
        bd.circleContainer.setBackgroundResource(bg);
    }

    private String scoreToLevel(int score) {
        if (score >= 70) return "high";
        if (score >= 35) return "medium";
        return "safe";
    }

    private void showFeedbackDialog() {
        new androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("反馈此次检测")
            .setItems(new String[]{"✅ 结果正确", "❌ 实际为正常号码 (误报)", "⚠️ 实际为诈骗号码 (漏报)"},
                (d, which) -> {
                    Toast.makeText(this, "感谢反馈，模型将持续优化", Toast.LENGTH_SHORT).show();
                    // TODO: 调用 /v1/infer/feedback
                })
            .setNegativeButton("取消", null)
            .show();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (streamClient != null) streamClient.cancel();
    }
}
