package com.campus.safety.ui.fragment;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;
import com.campus.safety.databinding.FragmentAlertsBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.FraudAlert;
import com.campus.safety.model.PageResult;
import com.campus.safety.network.ApiClient;
import com.campus.safety.ui.adapter.AlertAdapter;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class AlertsFragment extends Fragment {

    private FragmentAlertsBinding bd;
    private AlertAdapter adapter;
    private int currentPage = 1;
    private boolean isLoading = false;
    private boolean hasMore = true;

    @Override
    public View onCreateView(@NonNull LayoutInflater inf, ViewGroup c, Bundle s) {
        bd = FragmentAlertsBinding.inflate(inf, c, false);
        return bd.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View v, Bundle s) {
        LinearLayoutManager lm = new LinearLayoutManager(getContext());
        bd.rvAlerts.setLayoutManager(lm);
        adapter = new AlertAdapter(alert ->
            Toast.makeText(getContext(), alert.title, Toast.LENGTH_SHORT).show());
        bd.rvAlerts.setAdapter(adapter);
        bd.rvAlerts.setHasFixedSize(false);

        // 无限滚动
        bd.rvAlerts.addOnScrollListener(new RecyclerView.OnScrollListener() {
            @Override public void onScrolled(@NonNull RecyclerView rv, int dx, int dy) {
                if (dy <= 0) return;
                int last = lm.findLastVisibleItemPosition();
                if (!isLoading && hasMore && last >= adapter.getItemCount() - 3) {
                    loadAlerts(currentPage + 1);
                }
            }
        });

        // 下拉刷新
        bd.swipeRefresh.setColorSchemeResources(android.R.color.holo_blue_bright);
        bd.swipeRefresh.setOnRefreshListener(() -> {
            currentPage = 1;
            hasMore = true;
            adapter.clear();
            loadAlerts(1);
        });

        loadAlerts(1);
    }

    private void loadAlerts(int page) {
        if (isLoading) return;
        isLoading = true;
        bd.swipeRefresh.setRefreshing(page == 1);

        ApiClient.getApi().getAlerts(page, 20)
            .enqueue(new Callback<ApiResponse<PageResult<FraudAlert>>>() {
                @Override
                public void onResponse(Call<ApiResponse<PageResult<FraudAlert>>> c,
                                       Response<ApiResponse<PageResult<FraudAlert>>> r) {
                    if (!isAdded()) return;
                    isLoading = false;
                    bd.swipeRefresh.setRefreshing(false);
                    if (!r.isSuccessful() || r.body() == null || r.body().data == null) return;
                    PageResult<FraudAlert> pr = r.body().data;
                    if (page == 1) adapter.setItems(pr.items);
                    else adapter.appendItems(pr.items);
                    currentPage = page;
                    hasMore = pr.items != null && pr.items.size() >= 20;
                    bd.tvEmpty.setVisibility(adapter.getItemCount() == 0 ? View.VISIBLE : View.GONE);
                }
                @Override
                public void onFailure(Call<ApiResponse<PageResult<FraudAlert>>> c, Throwable t) {
                    if (!isAdded()) return;
                    isLoading = false;
                    bd.swipeRefresh.setRefreshing(false);
                }
            });
    }

    @Override public void onDestroyView() { super.onDestroyView(); bd = null; }
}
