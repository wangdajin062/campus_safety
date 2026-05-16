package com.campus.safety.ui.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;

import com.campus.safety.R;
import com.campus.safety.databinding.FragmentCallCheckBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.CallLog;
import com.campus.safety.model.PageResult;
import com.campus.safety.model.PhoneCheckResult;
import com.campus.safety.network.ApiClient;
import com.campus.safety.ui.activity.DetectionResultActivity;
import com.campus.safety.ui.activity.ReportActivity;
import com.campus.safety.ui.adapter.CallHistoryAdapter;

import java.util.regex.Pattern;

import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

/**
 * 来电检测 Fragment
 * ================
 * - 手机号输入（实时正则校验 + 格式化）
 * - 一键查询风险 → 跳转 DetectionResultActivity
 * - 一键举报
 * - 历史检测记录列表（分页）
 */
public class CallCheckFragment extends Fragment {

    private static final Pattern PHONE_RE = Pattern.compile("^1[3-9]\\d{9}$");
    private FragmentCallCheckBinding bd;
    private CallHistoryAdapter historyAdapter;
    private int currentPage = 1;
    private boolean loadingMore = false;

    @Override
    public View onCreateView(@NonNull LayoutInflater inf, ViewGroup c, Bundle s) {
        bd = FragmentCallCheckBinding.inflate(inf, c, false);
        return bd.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View v, Bundle s) {
        // 输入实时校验
        bd.etPhone.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void onTextChanged(CharSequence s, int a, int b, int c) {}
            @Override public void afterTextChanged(Editable e) {
                String phone = e.toString().trim();
                boolean valid = PHONE_RE.matcher(phone).matches();
                bd.btnCheck.setEnabled(valid);
                bd.btnReport.setEnabled(valid);
                if (phone.length() >= 11) bd.tilPhone.setError(valid ? null : "请输入正确的11位手机号");
                else bd.tilPhone.setError(null);
            }
        });

        bd.btnCheck.setOnClickListener(x -> checkPhone());
        bd.btnReport.setOnClickListener(x -> reportPhone());

        // 历史列表
        historyAdapter = new CallHistoryAdapter(log -> {
            Intent i = new Intent(getContext(), DetectionResultActivity.class)
                .putExtra("phone", log.phone)
                .putExtra("risk_score", log.risk_score)
                .putExtra("type", "history");
            startActivity(i);
        });
        bd.rvHistory.setAdapter(historyAdapter);

        // 上拉加载更多
        bd.rvHistory.addOnScrollListener(new androidx.recyclerview.widget.RecyclerView.OnScrollListener() {
            @Override public void onScrolled(@NonNull androidx.recyclerview.widget.RecyclerView rv, int dx, int dy) {
                if (dy <= 0 || loadingMore) return;
                androidx.recyclerview.widget.LinearLayoutManager lm =
                    (androidx.recyclerview.widget.LinearLayoutManager) rv.getLayoutManager();
                if (lm == null) return;
                if (lm.findLastVisibleItemPosition() >= historyAdapter.getItemCount() - 2) {
                    loadHistory(currentPage + 1);
                }
            }
        });

        bd.swipeRefresh.setOnRefreshListener(() -> {
            currentPage = 1;
            historyAdapter.clear();
            loadHistory(1);
        });

        loadHistory(1);
    }

    private void checkPhone() {
        String phone = bd.etPhone.getText().toString().trim();
        if (!PHONE_RE.matcher(phone).matches()) {
            Toast.makeText(getContext(), "请输入正确的手机号", Toast.LENGTH_SHORT).show();
            return;
        }

        bd.btnCheck.setEnabled(false);
        bd.btnCheck.setText("查询中...");

        ApiClient.getApi().checkPhone(phone).enqueue(new Callback<ApiResponse<PhoneCheckResult>>() {
            @Override
            public void onResponse(Call<ApiResponse<PhoneCheckResult>> c, Response<ApiResponse<PhoneCheckResult>> r) {
                if (!isAdded() || bd == null) return;
                bd.btnCheck.setText("查询风险");
                bd.btnCheck.setEnabled(true);
                if (!r.isSuccessful() || r.body() == null || !r.body().isSuccess() || r.body().data == null) {
                    toast("查询失败");
                    return;
                }
                PhoneCheckResult pr = r.body().data;
                Intent i = new Intent(getContext(), DetectionResultActivity.class)
                    .putExtra("phone", phone)
                    .putExtra("risk_level", pr.risk_level)
                    .putExtra("risk_score", pr.risk_score)
                    .putExtra("report_count", pr.report_count)
                    .putExtra("source", pr.source)
                    .putExtra("type", "phone_check");
                startActivity(i);
                loadHistory(1);  // 刷新历史
            }

            @Override
            public void onFailure(Call<ApiResponse<PhoneCheckResult>> c, Throwable t) {
                if (!isAdded() || bd == null) return;
                bd.btnCheck.setText("查询风险");
                bd.btnCheck.setEnabled(true);
                toast("网络错误：" + t.getMessage());
            }
        });
    }

    private void reportPhone() {
        String phone = bd.etPhone.getText().toString().trim();
        if (!PHONE_RE.matcher(phone).matches()) {
            Toast.makeText(getContext(), "请输入有效手机号", Toast.LENGTH_SHORT).show();
            return;
        }
        startActivity(new Intent(getContext(), ReportActivity.class).putExtra("phone", phone));
    }

    private void loadHistory(int page) {
        if (loadingMore) return;
        loadingMore = true;
        ApiClient.getApi().getCallHistory(page, 20).enqueue(new Callback<ApiResponse<PageResult<CallLog>>>() {
            @Override
            public void onResponse(Call<ApiResponse<PageResult<CallLog>>> c, Response<ApiResponse<PageResult<CallLog>>> r) {
                loadingMore = false;
                if (!isAdded() || bd == null) return;
                bd.swipeRefresh.setRefreshing(false);
                if (r.isSuccessful() && r.body() != null && r.body().isSuccess() && r.body().data != null) {
                    if (page == 1) historyAdapter.setItems(r.body().data.items);
                    else historyAdapter.appendItems(r.body().data.items);
                    currentPage = page;
                    bd.tvEmpty.setVisibility(historyAdapter.getItemCount() == 0 ? View.VISIBLE : View.GONE);
                }
            }

            @Override
            public void onFailure(Call<ApiResponse<PageResult<CallLog>>> c, Throwable t) {
                loadingMore = false;
                bd.swipeRefresh.setRefreshing(false);
            }
        });
    }

    private void toast(String m) {
        if (getContext() != null) Toast.makeText(getContext(), m, Toast.LENGTH_SHORT).show();
    }

    @Override
    public void onDestroyView() { super.onDestroyView(); bd = null; }
}
