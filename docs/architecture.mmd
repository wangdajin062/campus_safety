flowchart TB
    subgraph Android["Android Client (Java)"]
        direction TB
        SR[SmsReceiver] --> SFE[SmsFeatureExtractor]
        SFE --> QR[quickRisk local]
        SFE --> REQ[buildRequest]
        AEE[AcousticEmbedding\n128-dim F_v] --> REQ
        SD[SpeculativeDecoder\nDraft gamma=5] --> REQ
        OLE[OnDeviceLLMEngine] --> SD
        OLE --> NOTIF[Local Notification]
    end

    subgraph API["API Layer (FastAPI)"]
        EP[Inference Endpoints] --> RL[Rate Limit Redis]
        EP --> AUTH[JWT Auth]
        EP --> MD[MultimodalDetector]
    end

    subgraph Backend["Server (Python)"]
        MD --> RE[RuleEngine]
        MD --> GBM[GBM Classifier]
        MD --> FUSION[L-BFGS Fusion]
        FUSION --> RES[Risk Result]
        SD_VER[SpecVerify] --> RES
        QAD[QAD Pipeline] --> SD_VER
        AE[AcousticExtractor] --> EP
        FB[Feedback Logger] --> QAD
    end

    subgraph Data["Data Layer"]
        DB[(PostgreSQL)]
        RD[(Redis)]
        MDL[(GGUF Model\n240MB)]
    end

    REQ --> EP
    RES --> Android
    QAD --> MDL
    API --> DB
    API --> RD
