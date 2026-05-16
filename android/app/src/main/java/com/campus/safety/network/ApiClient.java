package com.campus.safety.network;

import android.content.Context;
import android.util.Log;

import com.campus.safety.BuildConfig;
import com.campus.safety.network.api.CampusApi;
import com.campus.safety.network.interceptor.AuthInterceptor;
import com.campus.safety.network.interceptor.ErrorInterceptor;
import com.campus.safety.util.TokenManager;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import okhttp3.logging.HttpLoggingInterceptor;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

/**
 * 网络客户端单例
 * ================
 * - 自动注入 JWT Bearer Token（AuthInterceptor）
 * - HTTP 401 时自动广播 ACTION_UNAUTHORIZED（ErrorInterceptor）
 * - Debug 模式开启 HTTP 日志
 * - 连接/读写超时 15s / 30s
 */
public class ApiClient {

    private static final String TAG = "ApiClient";
    private static Retrofit retrofit;
    private static OkHttpClient okHttpClient;
    private static Context appContext;

    public static void init(Context ctx) {
        appContext = ctx.getApplicationContext();

        HttpLoggingInterceptor logging = new HttpLoggingInterceptor(
            msg -> Log.d("OkHttp", msg)
        );
        logging.setLevel(BuildConfig.DEBUG_MODE ?
            HttpLoggingInterceptor.Level.BODY :
            HttpLoggingInterceptor.Level.NONE);

        okHttpClient = new OkHttpClient.Builder()
            .addInterceptor(new AuthInterceptor(appContext))
            .addInterceptor(new ErrorInterceptor(appContext))
            .addInterceptor(logging)
            .connectTimeout(15, TimeUnit.SECONDS)
            .readTimeout(30, TimeUnit.SECONDS)
            .writeTimeout(30, TimeUnit.SECONDS)
            .retryOnConnectionFailure(true)
            .build();

        retrofit = new Retrofit.Builder()
            .baseUrl(getBaseUrl())
            .client(okHttpClient)
            .addConverterFactory(GsonConverterFactory.create())
            .build();

        Log.i(TAG, "ApiClient initialized: " + getBaseUrl());
    }

    public static String getBaseUrl() {
        // Debug builds always use emulator-to-host tunnel (10.0.2.2:8888)
        // Release builds use the configured production URL
        return BuildConfig.API_BASE_URL;
    }

    public static CampusApi getApi() {
        if (retrofit == null) throw new IllegalStateException(
            "ApiClient not initialized. Call ApiClient.init(ctx) in Application.onCreate()");
        return retrofit.create(CampusApi.class);
    }

    public static OkHttpClient getOkHttpClient() { return okHttpClient; }

    /** 便捷方法：获取当前 JWT Token（供 SSE 等手动请求使用） */
    public static String getToken(Context ctx) {
        return TokenManager.getToken(ctx);
    }
}
