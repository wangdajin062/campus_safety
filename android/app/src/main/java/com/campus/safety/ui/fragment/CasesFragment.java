package com.campus.safety.ui.fragment;

import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.google.android.material.chip.Chip;
import com.campus.safety.databinding.FragmentCasesBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.FraudCase;
import com.campus.safety.model.PageResult;
import com.campus.safety.network.ApiClient;
import com.campus.safety.ui.activity.CaseDetailActivity;
import com.campus.safety.ui.adapter.CaseAdapter;
import java.util.Arrays;
import java.util.List;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class CasesFragment extends Fragment {

    private FragmentCasesBinding bd;
    private CaseAdapter adapter;
    private String currentCategory = null;
    private String currentKeyword = null;
    private int currentPage = 1;
    private boolean isLoading = false;
    private boolean hasMore = true;

    private static final List<String> CATEGORIES = Arrays.asList(
        "全部", "冒充公检法", "助学贷款", "兼职刷单",
        "虚假购物", "杀猪盘", "冒充客服", "其他"
    );

    @Override
    public View onCreateView(@NonNull LayoutInflater inf, ViewGroup c, Bundle s) {
        bd = FragmentCasesBinding.inflate(inf, c, false);
        return bd.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View v, Bundle s) {
        // 分类 Chip
        for (String cat : CATEGORIES) {
            Chip chip = new Chip(requireContext());
            chip.setText(cat);
            chip.setCheckable(true);
            chip.setChecked("全部".equals(cat));
            chip.setChipStrokeWidth(1f);
            chip.setTextSize(13f);
            chip.setOnCheckedChangeListener((btn, checked) -> {
                if (checked) {
                    currentCategory = "全部".equals(cat) ? null : cat;
                    reload();
                }
            });
            bd.chipGroup.addView(chip);
        }

        // RecyclerView
        LinearLayoutManager lm = new LinearLayoutManager(getContext());
        bd.rvCases.setLayoutManager(lm);
        adapter = new CaseAdapter(caseItem -> {
            Intent intent = new Intent(getActivity(), CaseDetailActivity.class);
            intent.putExtra("case_id", (long) caseItem.id);
            startActivity(intent);
        });
        bd.rvCases.setAdapter(adapter);
        bd.rvCases.setHasFixedSize(false);

        // 无限滚动
        bd.rvCases.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override public void onScrolled(@NonNull RecyclerView rv, int dx, int dy) {
                if (dy <= 0) return;
                int last = lm.findLastVisibleItemPosition();
                if (!isLoading && hasMore && last >= adapter.getItemCount() - 3) {
                    loadCases(currentPage + 1, false);
                }
            }
        });

        // 搜索
        bd.etSearch.addTextChangedListener(new TextWatcher() {
            @Override public void beforeTextChanged(CharSequence s, int st, int c, int a) {}
            @Override public void onTextChanged(CharSequence s, int st, int b, int c) {
                currentKeyword = s.length() >= 2 ? s.toString() : null;
                reload();
            }
            @Override public void afterTextChanged(Editable e) {}
        });

        // 下拉刷新
        bd.swipeRefresh.setColorSchemeResources(android.R.color.holo_blue_bright);
        bd.swipeRefresh.setOnRefreshListener(this::reload);

        loadCases(1, true);
    }

    private void reload() {
        currentPage = 1;
        hasMore = true;
        adapter.clear();
        loadCases(1, true);
    }

    private void showEmpty(String msg) {
        bd.tvEmpty.setVisibility(View.VISIBLE);
        bd.tvEmpty.setText(msg);
    }

    private void loadCases(int page, boolean showRefresh) {
        if (isLoading) return;
        isLoading = true;
        if (showRefresh) bd.swipeRefresh.setRefreshing(true);

        String cat = (currentCategory != null) ? currentCategory : "";
        String kw  = (currentKeyword  != null) ? currentKeyword  : "";
        ApiClient.getApi().getCases(
            cat.isEmpty() ? null : cat,
            kw.isEmpty()  ? null : kw,
            page, 15)
            .enqueue(new Callback<ApiResponse<PageResult<FraudCase>>>() {
                @Override
                public void onResponse(Call<ApiResponse<PageResult<FraudCase>>> c,
                                       Response<ApiResponse<PageResult<FraudCase>>> r) {
                    if (!isAdded()) return;
                    isLoading = false;
                    bd.swipeRefresh.setRefreshing(false);

                    // 详细错误检查，避免静默失败
                    if (!r.isSuccessful()) {
                        android.util.Log.e("CasesFragment",
                            "API error: " + r.code() + " " + r.message());
                        showEmpty("加载失败（" + r.code() + "），请稍后重试");
                        return;
                    }
                    if (r.body() == null) {
                        android.util.Log.e("CasesFragment", "Response body is null");
                        showEmpty("数据解析失败，请稍后重试");
                        return;
                    }
                    if (r.body().data == null) {
                        android.util.Log.e("CasesFragment",
                            "data is null, raw code=" + r.body().code);
                        showEmpty("暂无案例数据");
                        return;
                    }

                    PageResult<FraudCase> pr = r.body().data;
                    if (page == 1) adapter.setItems(pr.items);
                    else          adapter.appendItems(pr.items);
                    currentPage = page;
                    hasMore = pr.items != null && pr.items.size() >= 15;
                    bd.tvEmpty.setVisibility(
                        adapter.getItemCount() == 0 ? View.VISIBLE : View.GONE);
                    if (adapter.getItemCount() == 0)
                        bd.tvEmpty.setText("暂无相关案例");
                }
                @Override
                public void onFailure(Call<ApiResponse<PageResult<FraudCase>>> c, Throwable t) {
                    if (!isAdded()) return;
                    isLoading = false;
                    bd.swipeRefresh.setRefreshing(false);
                    android.util.Log.e("CasesFragment", "Network error: " + t.getMessage(), t);
                    showEmpty("网络连接失败，请检查网络后重试");
                }
            });
    }

    @Override public void onDestroyView() { super.onDestroyView(); bd = null; }
}
