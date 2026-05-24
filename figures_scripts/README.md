# QAD-MultiGuard — 论文图像复现包

本目录包含 QAD-MultiGuard 论文(v6,2026-05)中 **全部 10 张科学图像** 的可复现 Python 脚本与对应 PNG 输出。

> 论文:*QAD-MultiGuard: A Quantization-Aware Distilled Edge–Cloud Multimodal Framework for Privacy-Preserving Telecom Fraud Detection*

## 重要:数据来源(v6+ 单源真理设计)

从 v6 起,**所有图脚本均不内嵌数字常量**,而是统一从 `safety_data.py` 读取。这一模块逐字镜像项目代码库 `qad_multiguard/runs/*.json` 中的实验数据,确保:

1. **论文-代码-图三者完全同步**:同一份数据出现在 GitHub `runs/` JSON、`safety_data.py`、所有 fig*.py 与最终 docx 中。
2. **修改单点生效**:若 `runs/` 重新运行得到新数字,只需更新 `safety_data.py`,所有 10 张图自动重生成。
3. **来源可审计**:每个常量都标注其 JSON 来源(如 `EXP01_QUANT_QUALITY` 直接对应 `runs/exp01_quant_quality.json`)。

### `safety_data.py` 与 GitHub `runs/*.json` 的对应

| safety_data.py 常量 | GitHub runs 文件 | 用于 |
|------|------|------|
| `EXP01_QUANT_QUALITY`, `EXP01_F1_STD_PER_METHOD`, `QAT_QAD_OVF` | `exp01_quant_quality.json` + paper Table II | fig02 |
| `LATENCY_*`, `DEPLOYMENT`, `HEAD_TO_HEAD` | `exp02_end_to_end.json` | fig08 |
| `EXP03_LOSS_ABLATION` | `exp03_loss_ablation.json`(close 子集) | fig03 (a) |
| `EXP04_OVF_LAYER_ABLATION` | `exp04_ovf_ablation.json` + paper 扩展 | fig04 (a) |
| `EXP05_SPECULATIVE` | `exp05_speculative.json` | fig05 |
| `EXP06_PROGRESSIVE_F1`, `EXP06_FOLD_WEIGHTS`, `EXP06_ARCHITECTURE` | `exp06_fusion_cv.json` | fig06 |
| `EXP07_PRIVACY` | `exp07_privacy.json` | fig07 |
| `EXP09_TEACHER` | `exp09_teacher_selection.json` | fig03 (b) |
| `EXP10_OVF_STEP_RATIO` | `exp10_ovf_step_ratio.json` | fig04 (b) |

**注意**:`exp04_ovf_ablation.json` 包含 5 行(baseline/q_only/q_v/q_k_v/q_k_v_o),论文表 VI 额外引入了两个扩展点(FFN-only、q,k,v,o+FFN)。这两行在 `EXP04_OVF_LAYER_ABLATION` 中以 `"from_json": False` 标记,fig04 在视觉上以 **斜线纹** 区分,以保证数据来源可追溯。

## 目录结构

```
figures_scripts/
├── safety_data.py                     # ★ 单源真理(从 runs/*.json 同步)
├── sci_style.py                       # SCI 风格共享配置(色盲安全/300DPI)
├── fig01_architecture.py              # 图 1:三层端云架构(纯示意,无数字)
├── fig02_main_results.py              # 图 2:主结果 vs 11 基线(从 EXP01+QAT_QAD_OVF)
├── fig03_ablation_loss_teacher.py     # 图 3:损失/教师消融(从 EXP03+EXP09)
├── fig04_ovf_ablation.py              # 图 4:OV-Freeze 消融(从 EXP04+EXP10)
├── fig05_speculative.py               # 图 5:推测解码(从 EXP05)
├── fig06_fusion_analysis.py           # 图 6:多模态融合(从 EXP06)
├── fig07_privacy_glo.py               # 图 7:GLO 隐私验证(从 EXP07)
├── fig08_deployment.py                # 图 8:30 天部署(从 LATENCY_*/DEPLOYMENT/HEAD_TO_HEAD)
├── fig09_qad_pipeline.py              # 图 9:QAD 训练流水线(纯示意,无数字)
├── fig10_acoustic_embedding.py        # 图 10:F_v 构造(纯示意,无数字)
├── generate_all.py                    # 一键生成全部 10 张图(已测试,约 10 秒完成)
├── requirements.txt                   # matplotlib + numpy
├── README.md                          # 本文件
└── output/                            # 预生成 PNG(300 DPI)× 10
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或
pip install matplotlib>=3.5 numpy>=1.21
```

### 2. 验证 safety_data 完整性

```bash
python3 safety_data.py
# 应输出: "safety_data.py — all self-checks pass"
```

### 3. 一键生成全部图像

```bash
python3 generate_all.py
# → output/fig01_architecture.png
# → ... 共 10 张
```

约 10 秒完成。

### 4. 单独生成某张图

```bash
python3 fig02_main_results.py
```

## 数据同步流程(更新数字)

如果项目 `qad_multiguard/runs/` 重新运行得到新数字,按以下流程同步:

```bash
# 1. 运行项目复现脚本
cd qad_multiguard
python scripts/run_reproduction.py --quick

# 2. 查看新数字
cat runs/exp01_quant_quality.json

# 3. 更新 safety_data.py 中对应常量

# 4. 重新生成所有图
cd ../figures_scripts
python3 generate_all.py

# 5. 重新构建论文 docx(若需要)
```

**关键:不要在 fig*.py 中硬编码数字**。所有数字必须先进 `safety_data.py`,再被 fig 脚本引用。

## 与 GitHub `qad_multiguard/runs/` 的数值一致性证明

| 图 | safety_data 来源常量 | 验证方式 |
|------|------|------|
| fig02 | EXP01 + QAT_QAD_OVF | 与 paper Table II 行对应,EXP01 部分直接来自 `exp01_quant_quality.json` |
| fig03 (a) | EXP03_LOSS_ABLATION | 直接来自 `exp03_loss_ablation.json` 中 student="close" 的 5 个 loss 行 |
| fig03 (b) | EXP09_TEACHER | 直接来自 `exp09_teacher_selection.json` |
| fig04 (a) | EXP04_OVF_LAYER_ABLATION | 5 行直接从 `exp04_ovf_ablation.json`,2 行标记为 paper 扩展(斜线纹) |
| fig04 (b) | EXP10_OVF_STEP_RATIO | 直接来自 `exp10_ovf_step_ratio.json` |
| fig05 | EXP05_SPECULATIVE | 直接来自 `exp05_speculative.json`;曲线由公式 `speedup(α, γ)` 解析计算 |
| fig06 (a) | EXP06_PROGRESSIVE_F1 | 直接来自 `exp06_fusion_cv.json.progressive_f1` |
| fig06 (b) | EXP06_FOLD_WEIGHTS | exp06 fold_weights 经标准化(原始为 L-BFGS 含 bias) |
| fig06 (c) | EXP06_ARCHITECTURE | 直接来自 `exp06_fusion_cv.json.architecture_comparison` |
| fig07 | EXP07_PRIVACY | 直接来自 `exp07_privacy.json`,WB/BB 均 round 至 3 位小数 |
| fig08 (a) | LATENCY_P50_MS, LATENCY_P99_MS | 16+28+212+12=268, 22+36+268+16=342 与 paper §4.13 一致 |
| fig08 (b) | DEPLOYMENT | 直接来自 `exp02_end_to_end.json.deployment_metrics` |
| fig08 (c) | HEAD_TO_HEAD | 直接来自 `exp02_end_to_end.json.head_to_head` |

## 设计原则

每张图遵循以下 SCI 期刊规范:

| 维度 | 实现 |
|------|------|
| **分辨率** | 300 DPI |
| **字体** | DejaVu Sans(可移植) |
| **调色板** | 色盲安全(Wong 调色板) |
| **图像宽度** | 7.16 英寸(双栏) / 3.5 英寸(单栏) |
| **误差表示** | 5 次独立运行标准差 / 95% CI |
| **可读性** | 字号 ≥ 6.5pt(图例)、≥ 9pt(标题) |
| **本文方法高亮** | 橙色 `#ff7f0e` + 描边 `#cc5500` |
| **自明性** | 每张图独立可读,无需文字说明即可理解 |

## 字体兼容性

默认使用 DejaVu Sans(matplotlib 自带)。如需 Times New Roman 等系统字体,在 `sci_style.py` 中修改:

```python
mpl.rcParams.update({
    "font.family": "Times New Roman",
    ...
})
```

## 许可证

MIT(与论文主体一致)。

## 引用

```bibtex
@article{qad_multiguard_2026,
  title  = {QAD-MultiGuard: A Quantization-Aware Distilled Edge--Cloud
            Multimodal Framework for Privacy-Preserving Telecom Fraud Detection},
  author = {QAD-MultiGuard contributors},
  year   = {2026},
}
```
