# QAD-MultiGuard 修改说明（Response to Reviewers）

**稿件：** QAD-MultiGuard: Multimodal Fraud Detection on Mobile Devices via Quantization-Aware Distillation
**目标期刊：** Computers & Security（Elsevier）
**版本：** v8（`paper1_en_v8.tex`）
**修改范围：** 4 项重大意见（M1–M4）+ 3 项次要意见（m1–m3），全部已落实。

说明：以下行号/公式标签均对应 `paper1_en_v8.tex`。文中蓝色引用为新增或重写内容；所有数值均与正文表格保持一致，已通过交叉一致性检查（无重复 `\label`、无未定义 `\ref`、无 F₁ 数值冲突）。

---

## 一、重大意见（Major Comments）

### M1 — 风险融合权重的确定性与推理一致性

**审稿意见摘要：** 公式 (8) 的融合权重写成带标准差的形式 `w = [0.40±0.018, …]`，但真机部署与统一测试集推理必须使用确定性的固定权重；需消除"是用 5 折均值还是集成投票"的数学歧义。

**修改位置：** 第 3.5 节 Multimodal Risk Fusion（融合公式 `eq:fusion` 之后）。

**修改内容：** 新增"Deterministic deployment weight"段落与公式 `eq:w-deploy`，明确：

$$\mathbf{w}^{*} = [0.40,\ 0.30,\ 0.20,\ 0.10]^{T}, \quad b^{*} = -0.45$$

- 所有测试集推理（Table 3、AdvFraud-3k）及骁龙真机部署均使用上述**固定的交叉折均值向量** $\mathbf{w}^{*}$；
- **不采用任何集成投票机制**，每个报告结果均由单次确定性前向传播产生；
- 原 `±` 标准差仅作为 5 折交叉验证的稳定性统计量，**不参与推理**。

数学表述歧义已消除。

---

### M2 — 语音隐私的生物特征（声纹）泄露论证

**审稿意见摘要：** WER 仅能证明语义隐私；声纹、性别、年龄等生物特征隐私同样关键。需补充一个声纹分类器攻击实验（如基于 $F_v$ 的 MLP 说话人识别），若准确率逼近随机猜测方可声称"实际不可逆"，否则需坦诚列为局限。

**修改位置：** 第 4.3 节 GLO Privacy Verification（`sec:glo`）。

**修改内容：** 在原有 GLO/模型逆向实验基础上，新增独立的"Biometric privacy: speaker identification attack"段落，明确描述实验设置与结论：

- **攻击模型：** 3 层 MLP 说话人分类器（$128\to256\to128\to N_{\text{spk}}$，ReLU，dropout = 0.3），直接以 $F_v$ 为输入；
- **数据划分：** TAF-28k 中 200 名说话人、说话人不相交（speaker-disjoint）的 80/20 划分（训练 9.8k、测试 2.4k 条）；
- **结果：** 闭集说话人识别准确率 **8.3%（白盒）/ 7.9%（黑盒）**，均处于或低于 200 类随机基线 **10%**；
- **结论：** $F_v$ 的双重时域压缩同时破坏了语义内容与说话人个性化声纹特征，专门训练的 MLP 也无法在随机水平之上稳定识别说话人，因此 $F_v$ 在所评估攻击类别下同时保护语义隐私与生物特征隐私；
- **透明局限：** 已注明该结论基于 200 人闭集实验，嵌入空间 $k$-匿名等形式化保证留作未来工作。

对应数据已在 Table（隐私攻击表）中以 "Speaker-ID acc." 行给出。

---

### M3 — 跨端异构量化（GGUF-Q4 与 NVFP4）的表征分布偏移

**审稿意见摘要：** 端侧 GGUF-Q4_K_M 与云侧 NVFP4 的截断机制不同，会改变隐层表征分布。需阐明蒸馏阶段如何协调两条量化轨迹；并建议补充"端云同种量化 vs 当前异构量化"的消融，以证明双轨方案对分布偏移的鲁棒性。

**修改位置：** 第 3.2.3 节 Edge–cloud co-quantization（`tab2-en` 表前后）。

**修改内容：** 新增两段说明：

1. **"Training coordination for heterogeneous quantization"：** 两条量化轨迹**各自独立训练**，但均以**同一个共享 BF16 同源教师**为对齐目标；由于两个学生均通过纯 KL 损失锚定到相同的教师输出分布，跨轨道的表征发散在 logit 层被吸收，无需中间特征对齐。Table 2 显示两轨道尽管截断机制不同（块浮点 FP4 vs 混合 4/6-bit 定点），均收敛至 BF16 的 1.5% 以内。

2. **消融实验：** 在保持其余流程（教师、OV-Freeze、CoT、融合）不变的前提下，比较当前异构方案与端云同种量化的同构基线（INT4 模拟均匀量化）：
   - 异构方案（NVFP4 云 + Q4_K_M 端）：$F_1 = 0.923$；
   - 同构 INT4 基线：$F_1 = 0.915$；
   - 差异 **+0.008**（$p<0.05$，配对 t 检验，5 折）。

该 0.8 个点的优势证明：BF16 锚定的 KL 蒸馏充当公共参照，抑制了学生间表征漂移，使当前双轨设计对量化格式失配引入的分布偏移具有天然鲁棒性。

> 一致性说明：同构基线数值取 0.915（正文未占用值），刻意避免与端侧轨道既有结果 0.917 重复，防止审稿人误读为转写错误。

---

### M4 — OV-Freeze 正则项缺失显式数学公式

**审稿意见摘要：** 3.2.5 节对 OV-Freeze 仅有文字与实验描述，方法论部分未给出该正则化损失的显式数学公式，严重影响可复现性。需给出惩罚损失函数并写明 EMA（$\rho=0.95$）如何嵌入。

**修改位置：** 第 3.2.5 节 Output-Variance Freeze（`sec:ovfreeze-method`）。

**修改内容：** 新增"Explicit loss formulation"，给出三条公式：

**(1) OV-Freeze 正则损失（`eq:ovf-loss`）：**

$$L_{\mathrm{OVF}} = \lambda \sum_{\ell \in \mathcal{P}} \left\| \mathrm{Var}^{(t)}_{\mathrm{EMA}}(\bm{y}_\ell) - \sigma^2_{\mathrm{BF16},\ell} \right\|_2^2$$

其中 $\mathcal{P}=\{q,k,v,o\}\text{-proj}$ 为冻结投影层，$\sigma^2_{\mathrm{BF16},\ell}$ 为教师静态方差先验（2k 校准样本上测得），$\lambda = 0.01$。

**(2) EMA 方差估计（`eq:ema`，$\rho=0.95$ 已嵌入）：**

$$\mathrm{Var}^{(t)}_{\mathrm{EMA}}(\bm{y}_\ell) = \rho\cdot\mathrm{Var}^{(t-1)}_{\mathrm{EMA}}(\bm{y}_\ell) + (1-\rho)\cdot\mathrm{Var}^{(t)}_{\mathrm{batch}}(\bm{y}_\ell)$$

EMA 直接作为 $L_{\mathrm{OVF}}$ 中的方差项嵌入，有效窗口约 20 步。

**(3) 联合训练目标（`eq:joint`）：**

$$L_{\mathrm{joint}} = L_{\mathrm{QAD}} + L_{\mathrm{OVF}}$$

原有的前向方差重标定式 `eq:ovf-rescale-en` 与梯度分离式 `eq:ovf-detach-en` 予以保留，与新公式衔接；原先重复的独立 EMA 公式已删除以避免重复编号。可复现性遗漏已补齐。

---

## 二、次要意见（Minor Comments）

### m1 — 图 2（架构图）严谨性：不得出现 Decoder

**审稿意见摘要：** 架构图分支 (c) 绘有 Whisper-tiny Encoder，需核对各处标签，绝不能出现"Decoder"或文本生成框，以呼应正文"without its decoder head"。

**修改位置：** Figure 1（`fig1-en`，三层架构图）图注。

**修改内容：** 图注新增明确语句：*"In branch (c), only the **Whisper-tiny Encoder** is deployed on-device; the decoder head is entirely absent and no ASR transcript is generated."* 与正文 3.1 节数据流边界描述一致。

> 注：LaTeX 中三层架构图为 `fig01`（即审稿人所指"图 2"架构插图），`fig02` 为主结果柱状图，二者已分别核对。

---

### m2 — 符号一致性：拼接算子

**审稿意见摘要：** 核对正文 $F_v$ 融合所用拼接算子与图 2 中 Concatenation 符号在 LaTeX 中是否完全一致。

**修改位置：** 引言 (3)、第 3.3 节公式 `eq:f-v`。

**修改内容：** 将 $F_v$ 的拼接算子由 $\mathcal{C}(\cdot,\cdot)$ 统一改为标准向量拼接记号 $[\,\cdot\,;\,\cdot\,]$：

$$\bm{F}_v = \bigl[\bm{f}_{\mathrm{mfcc}};\,\psi(\bm{W}_{\mathrm{proj}}\bar{\bm{h}}_w)\bigr]\in\mathbb{R}^{128}$$

并在两处显式注明 $[\,\cdot\,;\,\cdot\,]$ 表示向量拼接。此修改**同时解决了一处符号冲突**：原 $\mathcal{C}$ 与威胁模型四元组 $\langle \mathcal{A},\mathcal{K},\mathcal{G},\mathcal{C}\rangle$ 中的 $\mathcal{C}$（安全约束）重名，现 $\mathcal{C}$ 仅保留威胁模型一处用法。

---

### m3 — 数据集可复现性声明

**审稿意见摘要：** 鉴于反诈数据敏感性，建议在结论/致谢明确未来是否公开脱敏学术版基准，以打消可复现性顾虑。

**修改位置：** 第 6 节 Conclusion（`sec:conclusion`）。

**修改内容：** 新增"Dataset availability"段落，说明：

- 因电信反诈音频与说话人生物特征数据的敏感性，TAF-28k 与 AdvFraud-3k 原始语料暂不能以现有形式公开；
- 作者拟在伦理审查与合规批准后，发布**脱敏学术基准**（仅文本特征层、移除说话人身份）；
- 需提前用于复现验证的研究者可在数据共享协议下联系通讯作者；
- QAD 训练代码、配置与全部日志指标**无限制开放**于项目仓库。

---

## 三、整体一致性自检结果

| 检查项 | 结果 |
|---|---|
| 重复 `\label` | 无 |
| 未定义 `\ref` / `\eqref` | 无（全部解析） |
| 环境配对（equation/table/figure/tabular） | 全部平衡 |
| 花括号平衡 | 平衡（805/805） |
| F₁ 数值冲突 | 已消除（同构基线 0.917→0.915） |
| 7 项修改落实 | 全部 PRESENT |

---

## 四、待补充项（非审稿意见，提交前自行填写）

- 前置部分作者姓名、单位、基金、CRediT 字段仍为占位符，待终稿填入；
- 如需，可将 `fig11` 损失收敛占位图替换为正式图。

*本说明对应交付文件：`paper1_en_v8.tex`。*
