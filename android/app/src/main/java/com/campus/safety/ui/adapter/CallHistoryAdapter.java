package com.campus.safety.ui.adapter;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.campus.safety.R;
import com.campus.safety.databinding.ItemCallLogBinding;
import com.campus.safety.ml.Utils;
import com.campus.safety.model.CallLog;
import java.util.ArrayList;
import java.util.List;

public class CallHistoryAdapter extends RecyclerView.Adapter<CallHistoryAdapter.VH> {
    public interface OnClick { void onClick(CallLog c); }
    private final List<CallLog> items = new ArrayList<>();
    private final OnClick onClick;
    public CallHistoryAdapter(OnClick oc) { this.onClick = oc; }
    public void setItems(List<CallLog> l) { items.clear(); if (l != null) items.addAll(l); notifyDataSetChanged(); }
    public void appendItems(List<CallLog> l) {
        if (l == null || l.isEmpty()) return;
        int s = items.size(); items.addAll(l); notifyItemRangeInserted(s, l.size());
    }
    public void clear() { int n = items.size(); items.clear(); notifyItemRangeRemoved(0, n); }

    @NonNull @Override public VH onCreateViewHolder(@NonNull ViewGroup p, int t) {
        return new VH(ItemCallLogBinding.inflate(LayoutInflater.from(p.getContext()), p, false));
    }
    @Override public void onBindViewHolder(@NonNull VH h, int pos) { h.bind(items.get(pos)); }
    @Override public int getItemCount() { return items.size(); }

    class VH extends RecyclerView.ViewHolder {
        final ItemCallLogBinding b;
        VH(ItemCallLogBinding bb) { super(bb.getRoot()); b = bb; }
        void bind(CallLog c) {
            b.tvPhone.setText(Utils.maskPhone(c.phone));
            b.tvScore.setText(String.valueOf(c.risk_score));
            b.tvTime.setText(c.checked_at);
            b.tvRiskLevel.setText("high".equals(c.risk_level) ? "高危" :
                                  "medium".equals(c.risk_level) ? "中危" : "安全");
            b.tvRiskLevel.setBackgroundResource(
                "high".equals(c.risk_level) ? R.drawable.bg_chip_red :
                "medium".equals(c.risk_level) ? R.drawable.bg_chip_orange : R.drawable.bg_chip_green);
            b.getRoot().setOnClickListener(v -> onClick.onClick(c));
        }
    }
}
