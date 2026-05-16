package com.campus.safety.network;

import android.content.Context;
import android.util.Log;

import com.campus.safety.model.CoTStreamEvent;
import com.campus.safety.model.MultimodalRequest;
import com.campus.safety.util.TokenManager;
import com.google.gson.Gson;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicBoolean;

import okhttp3.*;
import okhttp3.sse.EventSource;
import okhttp3.sse.EventSourceListener;
import okhttp3.sse.EventSources;

/**
 * SSE 流式推理客户端
 * ==================
 * 消费后端 /v1/infer/stream 端点的 Server-Sent Events：
 *   - fast_detection   — <40ms 快速检测结果
 *   - immediate_alert  — 高危立即预警
 *   - spec_draft       — 推测解码草稿统计
 *   - cot_stream       — 链式推理 token 流
 *   - final_result     — 最终融合结论
 *   - error / done     — 流结束标记
 */
public class StreamingInferenceClient {

    private static final String TAG = "StreamingInfer";
    private final Context ctx;
    private final Gson gson = new Gson();
    private EventSource currentSource;
    private final AtomicBoolean active = new AtomicBoolean(false);

    public StreamingInferenceClient(Context ctx) { this.ctx = ctx.getApplicationContext(); }

    public interface StreamCallback {
        void onFastResult(CoTStreamEvent event);
        void onImmediateAlert(CoTStreamEvent event);
        void onSpecDraft(CoTStreamEvent event);
        void onCotChunk(String chunk);
        void onFinalResult(CoTStreamEvent event);
        void onStreamError(String error);
        void onStreamComplete();
    }

    public void startStream(MultimodalRequest request, StreamCallback callback) {
        if (active.getAndSet(true)) {
            Log.w(TAG, "Stream already active, canceling previous");
            cancel();
        }

        String json = gson.toJson(request);
        String token = TokenManager.getToken(ctx);
        if (token == null) {
            callback.onStreamError("未登录");
            active.set(false);
            return;
        }

        Request req = new Request.Builder()
            .url(ApiClient.getBaseUrl() + "v1/infer/stream")
            .header("Authorization", "Bearer " + token)
            .header("Accept", "text/event-stream")
            .header("X-Client-Version", "3.0.0")
            .post(RequestBody.create(json, MediaType.get("application/json; charset=utf-8")))
            .build();

        EventSource.Factory factory = EventSources.createFactory(ApiClient.getOkHttpClient());
        currentSource = factory.newEventSource(req, new EventSourceListener() {

            @Override
            public void onEvent(EventSource source, String id, String type, String data) {
                try {
                    CoTStreamEvent evt = gson.fromJson(data, CoTStreamEvent.class);
                    if (evt == null) return;

                    String eventType = type != null ? type : (evt.event != null ? evt.event : "");

                    switch (eventType) {
                        case "fast_detection":
                            callback.onFastResult(evt);
                            break;
                        case "immediate_alert":
                            callback.onImmediateAlert(evt);
                            break;
                        case "spec_draft":
                            callback.onSpecDraft(evt);
                            break;
                        case "cot_stream":
                            if (evt.chunk != null) callback.onCotChunk(evt.chunk);
                            break;
                        case "final_result":
                            callback.onFinalResult(evt);
                            break;
                        case "done":
                            callback.onStreamComplete();
                            active.set(false);
                            break;
                        case "error":
                            callback.onStreamError(data);
                            active.set(false);
                            break;
                        default:
                            Log.v(TAG, "unhandled event: " + eventType);
                    }
                } catch (Exception e) {
                    Log.e(TAG, "parse error: " + e.getMessage());
                }
            }

            @Override
            public void onFailure(EventSource source, Throwable t, Response response) {
                String msg = t != null ? t.getMessage() : (response != null ? "HTTP " + response.code() : "unknown");
                Log.w(TAG, "Stream failure: " + msg);
                callback.onStreamError(msg);
                active.set(false);
            }

            @Override
            public void onClosed(EventSource source) {
                Log.d(TAG, "Stream closed");
                if (active.getAndSet(false)) callback.onStreamComplete();
            }
        });
    }

    public void cancel() {
        if (currentSource != null) {
            currentSource.cancel();
            currentSource = null;
        }
        active.set(false);
    }

    public boolean isActive() { return active.get(); }
}
