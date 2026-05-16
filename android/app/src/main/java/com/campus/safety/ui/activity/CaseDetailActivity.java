package com.campus.safety.ui.activity;

import android.os.Bundle;
import android.view.View;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import com.campus.safety.R;
import com.campus.safety.databinding.ActivityCaseDetailBinding;
import com.campus.safety.model.ApiResponse;
import com.campus.safety.model.FraudCase;
import com.campus.safety.network.ApiClient;
import java.util.Map;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class CaseDetailActivity extends AppCompatActivity {

    private ActivityCaseDetailBinding bd;
    private long caseId;
    private boolean isFavorited = false;

    @Override
    protected void onCreate(Bundle s) {
        super.onCreate(s);
        bd = ActivityCaseDetailBinding.inflate(getLayoutInflater());
        setContentView(bd.getRoot());

        // 返回按钮
        setSupportActionBar(bd.toolbar);
        if (getSupportActionBar() != null) {
            getSupportActionBar().setDisplayHomeAsUpEnabled(true);
            getSupportActionBar().setTitle("案例详情");
        }
        bd.toolbar.setNavigationOnClickListener(x -> finish());

        caseId = getIntent().getLongExtra("case_id", -1L);
        if (caseId < 0) { finish(); return; }

        bd.fabFavorite.setOnClickListener(x -> toggleFavorite());
        loadCase();
    }

    private void loadCase() {
        bd.progress.setVisibility(View.VISIBLE);
        ApiClient.getApi().getCaseDetail(caseId).enqueue(new Callback<ApiResponse<FraudCase>>() {
            @Override
            public void onResponse(Call<ApiResponse<FraudCase>> c,
                                   Response<ApiResponse<FraudCase>> r) {
                bd.progress.setVisibility(View.GONE);
                if (!r.isSuccessful() || r.body() == null || r.body().data == null) {
                    Toast.makeText(CaseDetailActivity.this, "加载失败，请重试", Toast.LENGTH_SHORT).show();
                    return;
                }
                render(r.body().data);
            }
            @Override
            public void onFailure(Call<ApiResponse<FraudCase>> c, Throwable t) {
                bd.progress.setVisibility(View.GONE);
                Toast.makeText(CaseDetailActivity.this, "网络连接失败", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void render(FraudCase c) {
        String titleText = (c.emoji != null ? c.emoji + " " : "") + (c.title != null ? c.title : "");
        bd.tvTitle.setText(titleText);
        bd.tvCategory.setText(c.category != null ? c.category : "");
        bd.tvViewCount.setText("👁 阅读 " + c.view_count);
        bd.tvContent.setText(c.content != null ? c.content : c.summary);
        bd.tvPublishTime.setText(c.published_at != null
            ? "发布：" + c.published_at.substring(0, 10) : "");
        isFavorited = c.is_favorited;
        updateFavIcon();
    }

    private void toggleFavorite() {
        ApiClient.getApi().toggleFavorite(caseId).enqueue(new Callback<ApiResponse<Map<String,Object>>>() {
            @Override
            public void onResponse(Call<ApiResponse<Map<String,Object>>> c, Response<ApiResponse<Map<String,Object>>> r) {
                if (r.isSuccessful()) {
                    isFavorited = !isFavorited;
                    updateFavIcon();
                    Toast.makeText(CaseDetailActivity.this,
                        isFavorited ? "已收藏" : "已取消收藏", Toast.LENGTH_SHORT).show();
                }
            }
            @Override
            public void onFailure(Call<ApiResponse<Map<String,Object>>> c, Throwable t) {
                Toast.makeText(CaseDetailActivity.this, "操作失败", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void updateFavIcon() {
        bd.fabFavorite.setImageResource(isFavorited ?
            R.drawable.ic_favorite_filled : R.drawable.ic_favorite_border);
    }
}
