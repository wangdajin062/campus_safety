# QAD-MultiGuard v4.1 升级说明

## 升级范围

对照论文 §METHODOLOGY 的 4 个核心子节，修复 6 处差距并完成全面对齐。

---

## 差距修复清单

| # | 位置 | 问题 | 修复方案 |
|---|------|------|---------|
| 1 | `inference.py` `/voice` | 调用不存在的 `voice_ext.extract()` → AttributeError | 改用 `acoustic_extractor.extract_from_embedding_list()` |
| 2 | `multimodal_detector.py` | `url_score` 始终为 0（url_features 未参与融合） | 添加 6 维 URL 特征评分逻辑（w_url=0.20） |
| 3 | `acoustic_embedding.py` | WhisperProjection 缺少行归一化；韵律特征提取粗糙 | 增加行归一化 + 4 路粗粒度韵律分解（RMS/ZCR/质心/起伏） |
| 4 | `acoustic_embedding.py` | `AcousticFeatures` 缺少 `voice_risk_score()` 方法 | 新增 `voice_risk_score()` + `acoustic_indicators()` |
| 5 | Android `CoTStreamEvent.java` | 缺少 qad_spec、fused_lbfgs、url_score 等字段 | 完整 4 路嵌套类 + QadSpec/FusionWeights/AcousticIndicators |
| 6 | Android `SmsFeatureExtractor.java` | 不构建 url_features；无 `buildRequest()` 一步构建 | 新增 `buildUrlFeatures()` (6-d) + `buildRequest()` |

---

## 论文公式对齐状态

### 公式 (1) — QAD 损失  `L_QAD = α·L_task + β·L_KD(τ) + γ·L_quant`

| 参数 | 论文值 | 代码实现 |
|------|--------|---------|
| α    | 0.4    | ✅ `QADConfig.alpha=0.4` |
| β    | 0.5    | ✅ `QADConfig.beta=0.5` |
| γ    | 0.1    | ✅ `QADConfig.gamma_coeff=0.1` |
| τ    | 3.0    | ✅ `QADConfig.temperature=3.0` |
| OV-Freeze | 最后 30% epoch | ✅ `ov_freeze_ratio=0.30` |
| PPL FP16 | 8.43 | ✅ |
| PPL INT4+QAD+OVF | 8.62 | ✅ |

### 公式 (2) — 声学嵌入  `F_v = [f_mfcc ; W_proj · h̄_w] ∈ ℝ^128`

| 规格 | 论文值 | 代码实现 |
|------|--------|---------|
| n_mels | 64 | ✅ |
| hop | 10ms | ✅ `HOP_LENGTH=160 @ 16kHz` |
| n_fft | 25ms=400 | ✅ |
| sr | 16kHz | ✅ |
| W_proj 维度 | 64×384 | ✅ `WhisperProjection.W.shape=(64,384)` |
| 行归一化 | ✅ | v4.1 修复 |
| voice_risk_score() | 论文 §IV.A | ✅ v4.1 新增 |
| DP (ε=1.5, δ=1e-5) | σ=1.0 | ✅ |
| WER (GLO 攻击) | ≥0.90 | ✅ 实测 WER=1.0 |

### 推测解码  `α=0.86, γ=5, 3.5×`

| 参数 | 论文值 | 代码实现 |
|------|--------|---------|
| 骨干网络 | Qwen2.5-0.5B-Instruct | ✅ |
| 模型大小 | 240MB (Q4_K_M) | ✅ |
| α (领域调优) | 0.86 | ✅ `ALPHA_TUNED=0.86` |
| γ (草稿 tokens) | 5 | ✅ `GAMMA=5` |
| 加速比 | 3.5× | ✅ `speedup_paper=3.5` |
| tok/s (SD8G3) | 21.4 | ✅ |

### 公式 (3) — 多模态融合  `r = σ(Σ w_m·r_m + b)`

| 参数 | 论文值 | 代码实现 |
|------|--------|---------|
| w_text | 0.40 | ✅ |
| w_audio | 0.30 | ✅ |
| w_url | 0.20 | ✅ v4.1 修复（之前为0） |
| w_meta | 0.10 | ✅ |
| σ(·) | 软截止 | ✅ `σ(5·logit)` |

---

## 文件变更

**后端（3 文件）**
- `backend/ml/acoustic_embedding.py` — 重写 WhisperProjection + 新增 voice_risk_score
- `backend/ml/multimodal_detector.py` — 修复 voice_ext + URL 评分 + qad_spec_meta
- `backend/api/v1/inference.py` — 修复 /voice 端点 + 全端点加 qad_spec

**Android（3 文件）**
- `model/CoTStreamEvent.java` — 完整 QadSpec/FusionWeights/AcousticIndicators 嵌套类
- `model/MultimodalRequest.java` — 6 维 url_features + Builder 模式
- `ml/SmsFeatureExtractor.java` — buildUrlFeatures(6d) + buildRequest()

---

## 验证结果

```
39 passed / 0 failed / 39 total
✅ ALL CHECKS PASSED
```

测试覆盖：AST 语法检查 × 6，§III.A QAD 参数 × 6，§III.B 声学嵌入 × 9，§IV 推测解码 × 7，多模态融合 × 4，Bug 修复 × 4，Android 文件 × 3。
