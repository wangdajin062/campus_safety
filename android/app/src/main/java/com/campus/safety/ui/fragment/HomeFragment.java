package com.campus.safety.ui.fragment;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import com.campus.safety.databinding.FragmentHomeBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.HomeData;
import com.campus.safety.network.ApiClient;
import com.campus.safety.ui.activity.MainActivity;
import com.campus.safety.ui.activity.ReportActivity;
import com.campus.safety.ui.adapter.AlertPreviewAdapter;
import com.campus.safety.util.TokenManager;
import android.content.Intent;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class HomeFragment extends Fragment {

    private FragmentHomeBinding bd;
    private AlertPreviewAdapter alertAdapter;

    @Override
    public View onCreateView(@NonNull LayoutInflater inf, ViewGroup c, Bundle s) {
        bd = FragmentHomeBinding.inflate(inf, c, false);
        return bd.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View v, Bundle s) {
        // 问候语
        String name = TokenManager.getNickname(requireContext());
        bd.tvGreeting.setText("Hi, " + (name != null && !name.isEmpty() ? name : "校园守护者"));

        // 预警预览 RecyclerView
        alertAdapter = new AlertPreviewAdapter(alert ->
            ((MainActivity) requireActivity()).switchToTab(3));
        bd.rvAlertsPreview.setLayoutManager(new LinearLayoutManager(getContext()));
        bd.rvAlertsPreview.setAdapter(alertAdapter);
        bd.rvAlertsPreview.setNestedScrollingEnabled(false);

        // 下拉刷新
        bd.swipeRefresh.setColorSchemeResources(
            android.R.color.holo_blue_bright,
            android.R.color.holo_green_light,
            android.R.color.holo_orange_light);
        bd.swipeRefresh.setOnRefreshListener(this::loadHome);

        // 快捷入口 — 查号
        bd.cardQuickCall.setOnClickListener(x ->
            ((MainActivity) requireActivity()).switchToTab(1));

        // 快捷入口 — 案例
        bd.cardQuickCases.setOnClickListener(x ->
            ((MainActivity) requireActivity()).switchToTab(2));

        // 快捷入口 — 举报（跳转到举报页）
        bd.cardQuickReport.setOnClickListener(x -> {
            Intent intent = new Intent(getActivity(), ReportActivity.class);
            startActivity(intent);
        });

        // 查看全部预警
        bd.tvMoreAlerts.setOnClickListener(x ->
            ((MainActivity) requireActivity()).switchToTab(3));

        loadHome();
    }

    private void loadHome() {
        bd.swipeRefresh.setRefreshing(true);
        ApiClient.getApi().getHomeData().enqueue(new Callback<ApiResponse<HomeData>>() {
            @Override
            public void onResponse(Call<ApiResponse<HomeData>> c,
                                   Response<ApiResponse<HomeData>> r) {
                if (!isAdded()) return;
                bd.swipeRefresh.setRefreshing(false);
                if (!r.isSuccessful() || r.body() == null || r.body().data == null) {
                    showError("数据加载失败，请下拉重试");
                    return;
                }
                bindData(r.body().data);
            }

            @Override
            public void onFailure(Call<ApiResponse<HomeData>> c, Throwable t) {
                if (!isAdded()) return;
                bd.swipeRefresh.setRefreshing(false);
                showError("网络连接异常，请检查网络");
            }
        });
    }

    private void bindData(HomeData d) {
        // 防护分数
        int score = 60;
        String level = "铜盾";
        String desc = "继续加强防护意识！";

        if (d.stats != null) {
            score = d.stats.protection_score > 0 ? d.stats.protection_score : 60;
            level = d.stats.protection_level != null ? d.stats.protection_level : "铜盾";
            if (score >= 80) desc = "防护优秀，您是校园安全卫士！";
            else if (score >= 60) desc = "防护良好，继续保持！";
            else desc = "建议多了解防骗知识，提升防护分数。";

            // 统计数字
            bd.tvBlockedCalls.setText(String.valueOf(d.stats.blocked_calls));
            bd.tvAlertedSms.setText(String.valueOf(d.stats.alerted_sms));
            bd.tvTotalReports.setText(String.valueOf(d.stats.total_reports));
            bd.tvCasesRead.setText(String.valueOf(d.stats.cases_read));
        } else {
            bd.tvBlockedCalls.setText(String.valueOf(d.blocked_today));
            bd.tvAlertedSms.setText(String.valueOf(d.alerted_sms));
            bd.tvTotalReports.setText("0");
            bd.tvCasesRead.setText("0");
        }

        bd.tvProtectionScore.setText(String.valueOf(score));
        bd.tvProtectionLevel.setText(level);
        bd.progressBar.setProgress(score);
        bd.tvScoreDesc.setText(desc);

        // 今日提示
        if (d.today_tip != null) {
            String tipTitle = (d.today_tip.emoji != null ? d.today_tip.emoji + " " : "💡 ")
                + (d.today_tip.title != null ? d.today_tip.title : "今日防骗提示");
            bd.tvTipTitle.setText(tipTitle);
            bd.tvTipContent.setText(d.today_tip.content != null ? d.today_tip.content : "");
            bd.cardTip.setVisibility(View.VISIBLE);
        }

        // 最新预警
        if (d.latest_alerts != null && !d.latest_alerts.isEmpty()) {
            alertAdapter.setItems(d.latest_alerts);
            bd.rvAlertsPreview.setVisibility(View.VISIBLE);
            bd.tvEmptyAlerts.setVisibility(View.GONE);
        } else {
            bd.rvAlertsPreview.setVisibility(View.GONE);
            bd.tvEmptyAlerts.setVisibility(View.VISIBLE);
        }
    }

    private void showError(String msg) {
        Toast.makeText(getContext(), msg, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        bd = null;
    }
}
