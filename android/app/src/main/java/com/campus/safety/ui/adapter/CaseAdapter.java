package com.campus.safety.ui.adapter;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.campus.safety.databinding.ItemCaseBinding;
import com.campus.safety.model.FraudCase;
import java.util.ArrayList;
import java.util.List;

public class CaseAdapter extends RecyclerView.Adapter<CaseAdapter.VH> {
    public interface OnClick { void onClick(FraudCase c); }
    private final List<FraudCase> items = new ArrayList<>();
    private final OnClick onClick;
    public CaseAdapter(OnClick oc) { this.onClick = oc; }
    public void setItems(List<FraudCase> l) { items.clear(); if (l != null) items.addAll(l); notifyDataSetChanged(); }
    public void appendItems(List<FraudCase> l) {
        if (l == null || l.isEmpty()) return;
        int s = items.size(); items.addAll(l); notifyItemRangeInserted(s, l.size());
    }
    public void clear() { int n = items.size(); items.clear(); notifyItemRangeRemoved(0, n); }

    @NonNull @Override public VH onCreateViewHolder(@NonNull ViewGroup p, int t) {
        return new VH(ItemCaseBinding.inflate(LayoutInflater.from(p.getContext()), p, false));
    }
    @Override public void onBindViewHolder(@NonNull VH h, int pos) { h.bind(items.get(pos)); }
    @Override public int getItemCount() { return items.size(); }

    class VH extends RecyclerView.ViewHolder {
        final ItemCaseBinding b;
        VH(ItemCaseBinding bb) { super(bb.getRoot()); b = bb; }
        void bind(FraudCase c) {
            b.tvTitle.setText((c.emoji != null ? c.emoji + " " : "") + c.title);
            b.tvSummary.setText(c.summary);
            b.tvCategory.setText(c.category);
            b.tvViewCount.setText("👁 " + c.view_count);
            b.tvFeatured.setVisibility(c.is_featured ? View.VISIBLE : View.GONE);
            b.getRoot().setOnClickListener(v -> onClick.onClick(c));
        }
    }
}
