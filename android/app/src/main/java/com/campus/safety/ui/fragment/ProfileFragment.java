package com.campus.safety.ui.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.view.*;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import com.campus.safety.databinding.FragmentProfileBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.UserStats;
import com.campus.safety.network.ApiClient;
import com.campus.safety.ui.activity.LoginActivity;
import com.campus.safety.ui.activity.SettingsActivity;
import com.campus.safety.util.TokenManager;
import com.campus.safety.ml.Utils;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 我的 Fragment
 * - 头像/昵称/手机号
 * - 防护等级/分数
 * - 4 维统计
 * - 设置入口
 * - 退出登录
 */
public class ProfileFragment extends Fragment {
    private FragmentProfileBinding bd;

    @Override
    public View onCreateView(@NonNull LayoutInflater inf, ViewGroup c, Bundle s) {
        bd = FragmentProfileBinding.inflate(inf, c, false);
        return bd.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View v, Bundle s) {
        bd.tvNickname.setText(TokenManager.getNickname(requireContext()));
        bd.tvPhone.setText(Utils.maskPhone(TokenManager.getPhone(requireContext())));
        bd.btnSettings.setOnClickListener(x -> startActivity(new Intent(getContext(), SettingsActivity.class)));
        bd.btnLogout.setOnClickListener(x -> confirmLogout());
        bd.swipeRefresh.setOnRefreshListener(this::loadStats);
        loadStats();
    }

    private void loadStats() {
        bd.swipeRefresh.setRefreshing(true);
        ApiClient.getApi().getUserStats().enqueue(new Callback<ApiResponse<UserStats>>() {
            @Override public void onResponse(Call<ApiResponse<UserStats>> c, Response<ApiResponse<UserStats>> r) {
                if (!isAdded() || bd == null) return;
                bd.swipeRefresh.setRefreshing(false);
                if (!r.isSuccessful() || r.body() == null || !r.body().isSuccess() || r.body().data == null) return;
                UserStats us = r.body().data;
                bd.tvProtectionScore.setText(String.valueOf(us.protection_score));
                bd.tvProtectionLevel.setText(us.protection_level != null ? us.protection_level :
                    (us.protection_score >= 90 ? "钻石盾" : us.protection_score >= 75 ? "金盾" :
                     us.protection_score >= 60 ? "银盾" : "铜盾"));
                bd.progressProtection.setProgress(us.protection_score);
                bd.tvBlockedCalls.setText(String.valueOf(us.blocked_calls));
                bd.tvAlertedSms.setText(String.valueOf(us.alerted_sms));
                bd.tvTotalReports.setText(String.valueOf(us.total_reports));
                bd.tvCasesRead.setText(String.valueOf(us.cases_read));
            }
            @Override public void onFailure(Call<ApiResponse<UserStats>> c, Throwable t) {
                if (!isAdded() || bd == null) return;
                bd.swipeRefresh.setRefreshing(false);
            }
        });
    }

    private void confirmLogout() {
        new androidx.appcompat.app.AlertDialog.Builder(requireContext())
            .setTitle("退出登录")
            .setMessage("确定要退出当前账号吗？")
            .setPositiveButton("退出", (d, w) -> {
                TokenManager.clear(requireContext());
                startActivity(new Intent(getContext(), LoginActivity.class)
                    .setFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK));
                requireActivity().finish();
            })
            .setNegativeButton("取消", null)
            .show();
    }

    @Override public void onDestroyView() { super.onDestroyView(); bd = null; }
}
