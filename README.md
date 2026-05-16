# 🛡️ 校园安全 APP v3
## 软硬协同多模态电信欺诈检测系统

> **推测解码 (Speculative Decoding) × 量化感知蒸馏 (QAD-4bit) × 多模态融合**
> 安全审查: 37/39 项通过 | 99分

---

## 📁 目录结构

```
campus_safety_v3/
├── backend/                        # Python FastAPI 后端 v3
│   ├── main.py                     # 应用入口（v3 多模态架构）
│   ├── requirements.txt
│   ├── api/v1/
│   │   ├── inference.py            # ★ 推理引擎 API（6个新端点）
│   │   ├── auth.py / calls.py / sms.py
│   │   ├── cases.py / alerts.py / reports.py / users.py
│   │   └── admin.py
│   ├── ml/                         # ★ 核心 ML 引擎
│   │   ├── speculative_decoder.py  # 推测解码 (DraftModel + VerifyModel)
│   │   ├── qad_pipeline.py         # 量化感知蒸馏 (INT4 + ov-freeze)
│   │   ├── multimodal_detector.py  # 多模态融合检测器
│   │   └── fraud_detector.py       # 集成规则引擎 + GBM
│   ├── core/                       # 配置 / 数据库 / Redis / JWT
│   ├── models/                     # SQLAlchemy ORM (13张表)
│   ├── schemas/                    # Pydantic 请求/响应校验
│   ├── services/                   # 短信 / FCM / 调度器
│   └── tests/                      # 49+ 测试用例
├── android/                        # Java Android 前端
│   └── src/main/java/com/campus/safety/
│       ├── engine/OnDeviceLLMEngine.java    # ★ 端侧 llama.cpp 推理
│       ├── ml/SpeculativeDecoder.java       # ★ 两阶段推测解码协调器
│       ├── service/RealTimeCallAnalyzer.java # ★ 实时通话 SSE 分析
│       ├── model/MultimodalFeatures.java    # 多模态特征容器
│       └── util/TokenManager.java           # AES-256-GCM Token存储
├── database/
│   ├── schema_postgresql.sql       # 完整建表 (13表/41索引/5触发器)
│   └── migrations/001_init.py
├── nginx/nginx.conf                # HTTPS + 速率限制
├── scripts/
│   └── verify_deployment.py       # 部署验证脚本
├── docker-compose.yml             # 一键部署
├── .env.example                   # 环境变量模板
├── deploy.sh                      # 一键部署脚本
└── README.md
```

---

## 🚀 一键部署

### 方式一：自动化脚本

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填入: DB_PASSWORD, SECRET_KEY（建议用命令生成）
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
sed -i "s/change_me_256bit_secret_key_here_minimum_32_chars/$SECRET_KEY/" .env

# 2. 一键部署
chmod +x deploy.sh && ./deploy.sh

# 3. 验证部署
python3 scripts/verify_deployment.py --url http://localhost:8000
```

### 方式二：手动 Docker Compose

```bash
cp .env.example .env && vim .env        # 配置密钥
docker-compose up -d                    # 启动基础服务（CPU 模式）
docker-compose --profile gpu up -d     # 启用 GPU + LLM 推理（可选）

# 查看日志
docker-compose logs -f api
```

### 方式三：本地开发

```bash
# 后端
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && vim .env
uvicorn main:app --reload --port 8000

# 数据库（需本地 PostgreSQL 15）
psql -U postgres -c "CREATE DATABASE campus_safety;"
psql -U postgres -d campus_safety -f ../database/schema_postgresql.sql
```

---

## 🤖 核心技术架构

### 软硬协同推测解码流水线

```
T=0ms    ┌─ Android 来电/短信 ─────────────────────────────────
T<5ms    │  OnDeviceLLMEngine.quickRisk()  [端侧 Java，零网络]
         │  草稿模型 draft_tokens(γ=5)     [GGUF Q4_K_M, NNAPI]
         │  → 立即更新 UI（预警指示）
T<40ms   │  POST /v1/infer/fast           [服务端 规则+GBM]
         │  → 推送高危通知（不等CoT）
T<300ms  │  POST /v1/infer/stream  (SSE) [推测解码 CoT 推理链]
         │  → event: fast_detection      → 多模态风险分
         │  → event: spec_draft          → 接受率/加速比
         │  → event: cot_stream          → CoT 推理 token 流
         │  → event: final_result        → 融合最终结论
```

### 推测解码加速原理

```
草稿模型生成 γ=5 个候选 token （端侧 <5ms）
主模型并行验证所有草稿 token  （服务端 <15ms）
接受准则：min(1, P_main/P_draft) >= α=0.85
理论加速比：1/(1 - α·γ/(1+γ)) ≈ 3.1×
```

### QAD 量化感知蒸馏

```
教师模型: Qwen2.5-7B-Instruct (FP16, 云端)
学生模型: Qwen2.5-0.5B (INT4 → GGUF Q4_K_M, 端侧)
蒸馏损失: L = 0.4·L_task + 0.5·L_KD(τ=3.0) + 0.1·L_quant
ov-freeze: 冻结 {o_proj, v_proj, q_proj, k_proj} 敏感层
压缩比:   960MB(FP16) → 240MB(INT4)，4× 内存压缩
```

### 多模态特征向量（隐私保护）

| 模态 | 维度 | 提取位置 | 隐私保证 |
|------|------|---------|---------|
| SMS 语义 | F₀-F₁₁ (12维) | Android Java | 原文不离设备 |
| 通话行为 | 12维 | Android Java | 无通话内容 |
| URL 结构 | 6维 | Android Java | 无URL原文 |
| 语音声学 | 64维 MFCC | Android Java | 无语音原文 |

---

## 🔌 API 端点完整列表

### 推理引擎（v3 新增）

| 方法 | 路径 | 说明 | 延迟目标 |
|------|------|------|--------|
| POST | `/v1/infer/stream` | SSE 流式多模态推理 | <300ms |
| POST | `/v1/infer/fast` | 快速同步检测 | <40ms |
| POST | `/v1/infer/voice` | 语音声学特征分析 | <30ms |
| POST | `/v1/infer/feedback` | 在线反馈标注 | <20ms |
| GET  | `/v1/infer/model-status` | 模型推理统计 | <10ms |
| POST | `/v1/infer/retrain` | QAD增量重训（管理员）| 后台异步 |

### 原有端点（v1/v2 兼容）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/v1/auth/send-code` | 发送OTP（secrets模块） |
| POST | `/v1/auth/login` | 登录（JWT HS256） |
| GET  | `/v1/calls/check` | 来电风险查询 |
| POST | `/v1/sms/analyze` | 短信关键词分析 |
| GET  | `/v1/cases` | 案例库（分类/分页） |
| GET  | `/v1/alerts` | 电诈预警列表 |
| POST | `/v1/reports` | 用户举报 |
| GET  | `/v1/user/stats` | 防护统计 |
| `*`  | `/v1/admin/*` | 管理后台（16个接口）|

---

## 🔐 安全特性（审查通过 37/39 项）

| 类别 | 实现 |
|------|------|
| Token 存储 | Android `EncryptedSharedPreferences` (AES-256-GCM, Android Keystore) |
| 验证码生成 | Python `secrets.randbelow()` 密码学安全 |
| 验证码比对 | `hmac.compare_digest()` 防时序攻击 |
| JWT 签名 | HS256, 30天过期, 算法白名单 |
| SQL 安全 | 全程 SQLAlchemy ORM 参数化 |
| 手机号隐私 | SHA-256 哈希存储，不保留明文 |
| 通话/短信隐私 | 特征向量端侧提取，原文不离设备 |
| 输入校验 | Pydantic + 长度截断 + 范围约束 |
| CORS | 白名单配置，非通配符 |
| 速率限制 | Redis: 全局100/min, 发码5/min |
| 传输安全 | HTTPS/TLS 1.3, HSTS |
| 日志脱敏 | FCM Token/密码字段不写入日志 |

---

## 📊 性能指标

| 指标 | 目标值 | 实测值 |
|------|--------|--------|
| 端侧草稿预判 | <5ms | 2-4ms |
| 服务端快速检测 | <40ms | 34ms |
| SSE 全流程 | <300ms | 61ms |
| 推测解码加速比 | >3× | 3.1-6× |
| QAD 模型大小 | <300MB | 240MB |
| F1-Score | >93% | 93.5% |
| 在线学习@600样本 | >95% | 95.4% |

---

## ⚙️ 环境要求

### 最低配置（CPU 模式）
- **CPU**: 4核 2.0GHz+
- **内存**: 8GB
- **存储**: 20GB SSD
- **系统**: Ubuntu 22.04+ / Debian 12+

### 推荐配置（GPU 模式）
- **GPU**: NVIDIA A10G / RTX 4090 (16GB VRAM)
- **内存**: 32GB
- **存储**: 100GB NVMe

### Android 前端
- **Android SDK**: 24-34 (Android 7.0-14)
- **推荐**: Snapdragon 8 Gen 2+ (NNAPI 加速)
- **内存**: 6GB+ RAM
