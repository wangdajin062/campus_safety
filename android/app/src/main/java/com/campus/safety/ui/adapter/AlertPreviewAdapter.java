package com.campus.safety.ui.adapter;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;
import com.campus.safety.databinding.ItemAlertPreviewBinding;
import com.campus.safety.model.FraudAlert;
import java.util.ArrayList;
import java.util.List;

public class AlertPreviewAdapter extends RecyclerView.Adapter<AlertPreviewAdapter.VH> {

    public interface OnClick { void onClick(FraudAlert a); }

    private final List<FraudAlert> items = new ArrayList<>();
    private final OnClick onClick;

    public AlertPreviewAdapter(OnClick oc) { this.onClick = oc; }

    public void setItems(List<FraudAlert> l) {
        items.clear();
        if (l != null) items.addAll(l);
        notifyDataSetChanged();
    }

    @NonNull @Override
    public VH onCreateViewHolder(@NonNull ViewGroup p, int t) {
        return new VH(ItemAlertPreviewBinding.inflate(LayoutInflater.from(p.getContext()), p, false));
    }

    @Override
    public void onBindViewHolder(@NonNull VH h, int pos) { h.bind(items.get(pos)); }

    @Override public int getItemCount() { return items.size(); }

    class VH extends RecyclerView.ViewHolder {
        final ItemAlertPreviewBinding b;
        VH(ItemAlertPreviewBinding bb) {
            super(bb.getRoot());
            b = bb;
        }
        void bind(FraudAlert a) {
            b.tvEmoji.setText(a.emoji != null ? a.emoji : "⚠️");
            b.tvTitle.setText(a.title != null ? a.title : "");
            b.tvContent.setText(a.content != null ? a.content : "");
            b.tvUrgentBadge.setVisibility(a.is_urgent ? View.VISIBLE : View.GONE);
            b.getRoot().setOnClickListener(v -> onClick.onClick(a));
        }
    }
}
