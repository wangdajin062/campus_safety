package com.campus.safety.ui.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.campus.safety.databinding.ItemAlertBinding;
import com.campus.safety.model.FraudAlert;
import java.util.ArrayList;
import java.util.List;

public class AlertAdapter extends RecyclerView.Adapter<AlertAdapter.VH> {

    public interface OnClick { void onClick(FraudAlert a); }

    private final List<FraudAlert> items = new ArrayList<>();
    private final OnClick onClick;

    public AlertAdapter(OnClick oc) { this.onClick = oc; }

    public void setItems(List<FraudAlert> l) {
        items.clear();
        if (l != null) items.addAll(l);
        notifyDataSetChanged();
    }

    public void appendItems(List<FraudAlert> l) {
        if (l == null || l.isEmpty()) return;
        int s = items.size();
        items.addAll(l);
        notifyItemRangeInserted(s, l.size());
    }

    public void clear() {
        int n = items.size();
        items.clear();
        notifyItemRangeRemoved(0, n);
    }

    @NonNull @Override
    public VH onCreateViewHolder(@NonNull ViewGroup p, int t) {
        return new VH(ItemAlertBinding.inflate(LayoutInflater.from(p.getContext()), p, false));
    }

    @Override public void onBindViewHolder(@NonNull VH h, int pos) { h.bind(items.get(pos)); }
    @Override public int getItemCount() { return items.size(); }

    class VH extends RecyclerView.ViewHolder {
        final ItemAlertBinding b;
        VH(ItemAlertBinding bb) {
            super(bb.getRoot());
            b = bb;
        }
        void bind(FraudAlert a) {
            b.tvEmoji.setText(a.emoji != null ? a.emoji : "⚠️");
            b.tvTitle.setText(a.title != null ? a.title : "");
            b.tvContent.setText(a.content != null ? a.content : "");
            b.tvUrgent.setVisibility(a.is_urgent ? View.VISIBLE : View.GONE);

            // 风险等级显示
            if ("high".equals(a.risk_level)) {
                b.tvRiskLevel.setText("🔴 高危");
            } else if ("medium".equals(a.risk_level)) {
                b.tvRiskLevel.setText("🟠 中危");
            } else {
                b.tvRiskLevel.setText("🟢 低危");
            }

            // 时间格式化
            String time = a.relative_time != null ? a.relative_time :
                         (a.published_at != null ? a.published_at.substring(0, 10) : "");
            b.tvTime.setText(time);

            b.getRoot().setOnClickListener(v -> onClick.onClick(a));
        }
    }
}
