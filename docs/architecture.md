%% QAD-MultiGuard v4.1 软硬协同多模态检测架构
```mermaid
flowchart TB
    subgraph Android["📱 Android 端侧 (Java)"]
        direction TB
        SR[SmsReceiver<br/>广播接收] --> SFE[SmsFeatureExtractor<br/>Features.extract()]
        SFE --> QR[quickScore/quickRiskLevel<br/>&lt;5ms 本地预警]
        SFE -- 12维 SMS 特征 --> REQ[MultimodalRequest<br/>buildRequest()]
        SFE -- 6维 URL 特征 --> REQ
        AEE[AcousticEmbeddingExtractor<br/>128维 F_v] --> REQ
        OLE[OnDeviceLLMEngine] --> SD[SpeculativeDecoder<br/>generateDraft γ=5]
        SD -- α=0.86 统计先验 --> REQ
        OLE -- quickRisk 关键词评分 --> NOTIF[NotificationHelper<br/>高危立即通知]
    end

    subgraph API["🌐 API 层 (FastAPI)"]
        direction TB
        EP[/v1/infer/fast<br/>/v1/infer/stream<br/>/v1/infer/voice] --> RL[rate_limit<br/>Redis 限流]
        EP --> AUTH[JWT 鉴权<br/>get_current_user]
        EP --> MD[MultimodalDetector]
    end

    subgraph Backend["🖥️ 服务端 (Python)"]
        direction TB
        MD --> RE[RuleEngine<br/>关键词规则]
        MD --> GBM[GradientBoosting<br/>GBM 分类器]
        MD --> FUSION[多模态融合<br/>L-BFGS σ(Σ w_m·r_m)]
        FUSION --> RES{{risk_level risk_score<br/>url_score voice_score}}
        SD_VER[SpeculativeDecoder.verify<br/>服务端验证] --> RES
        QAD[QAD Pipeline<br/>L=0.4L_task+0.5L_KD+0.1L_quant] --> SD_VER
        AE[AcousticExtractor<br/>voice_risk_score<br/>4路韵律分解] --> EP
        FB[record_feedback<br/>feedback.jsonl] --> QAD
    end

    subgraph Data["🗄️ 数据层"]
        DB[(PostgreSQL<br/>13张表)]
        RD[(Redis<br/>验证码/限流/黑名单)]
        MDL[fraud_draft_q4km.gguf<br/>Qwen2.5-0.5B Q4_K_M]
    end

    REQ -- HTTP/JSON --> EP
    RES --> Android
    QAD --> MDL
    API --> DB
    API --> RD
```
