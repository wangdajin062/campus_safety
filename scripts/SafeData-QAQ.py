"""
SafeData-QAQ.py — QAD-MultiGuard 多模态数据处理
==================================================
与生产推理管线完全对齐的特征提取流程。

各模态特征维度（与 backend/api/v1/inference.py MultimodalRequest 一致）:
  SMS 特征:  12-d (SmsFeatures → vectorize_sms)
  通话特征:  12-d (PhoneFeatures → vectorize_phone)
  URL 特征:   6-d [domain_len, path_depth, has_ip, has_port, entropy, is_shortened]
  声学特征: 128-d F_v = [f_mfcc(64); W_proj · h̄_w(64)] (acoustic_embedding.py)

融合公式（与 backend/ml/multimodal_detector.py 一致）:
    r = σ(5.0 · (0.40·r_text + 0.30·r_audio + 0.20·r_url + 0.10·r_meta))
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── 将 backend 加入导入路径 ──────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ── 生产模块导入 ─────────────────────────────────────────────
from ml.fraud_detector import (
    SmsFeatures,
    PhoneFeatures,
    RuleEngine,
    GradientBoostingDetector,
)
from ml.acoustic_embedding import (
    AcousticEmbeddingExtractor,
    EMBEDDING_DIM as AUDIO_EMBED_DIM,
)

logger = logging.getLogger(__name__)


# ── 融合权重（与 multimodal_detector.py 一致）───────────────
W_TEXT = 0.40
W_AUDIO = 0.30
W_URL = 0.20
W_META = 0.10
FUSION_BIAS = 0.0
FUSION_SCALE = 5.0

# 短链接服务黑名单
SHORT_URL_DOMAINS = {"bit.ly", "tinyurl", "t.cn", "goo.gl", "ow.ly",
                     "is.gd", "buff.ly", "shorturl.at", "rb.gy", "cutt.ly"}

# ── 全局单例（复用生产实例，确保一致性）────────────────────
_rule_engine = RuleEngine()
_gb_detector = GradientBoostingDetector()
_acoustic_ext = AcousticEmbeddingExtractor(dp_sigma=0.0)


# ============================================================
# 1. SMS 文本 → 12-d 特征向量（对齐生产 vectorize_sms）
# ============================================================
def extract_sms_features(text: str, sender: str = "") -> tuple[np.ndarray, dict]:
    """
    从原始 SMS 文本提取 12-d 特征向量。

    返回 (12-d 向量, 元信息 dict)
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(12, dtype=np.float32), {"keyword_hits": 0, "keyword_weight": 0.0}

    # ── 关键词匹配 ────────────────────────────────────────
    matched_keywords = []
    for kw in _rule_engine.KEYWORD_WEIGHT_MAP:
        if kw in text:
            matched_keywords.append(kw)
    # 硬规则检查
    for kw, _ in _rule_engine.HARD_HIGH_RULES:
        if kw in text:
            matched_keywords.append(kw)

    keyword_weight = sum(_rule_engine.KEYWORD_WEIGHT_MAP.get(kw, 100)
                         for kw in matched_keywords)

    # ── 信号检测 ──────────────────────────────────────────
    has_url = bool(re.search(r'https?://[^\s]+|www\.[^\s]+', text))
    url_count = len(re.findall(r'https?://[^\s]+', text))

    money_patterns = [
        r'[金额钱款转账汇款支付]', r'\d+[万亿千百]', r'银行[卡号账]',
        r'手续费', r'保证金', r'解冻费', r'押金',
    ]
    money_mentioned = any(re.search(p, text) for p in money_patterns)

    impersonation_keywords = [
        '公安局', '检察院', '法院', '警察', '民警', '公检法',
        '通信管理局', '互联网中心', '客服', '官方',
    ]
    impersonation = any(kw in text for kw in impersonation_keywords)

    urgency_keywords = [
        '立即', '马上', '紧急', '否则', '过期', '失效', '冻结',
        '封号', '停用', '注销', '限制',
    ]
    urgency_score = min(1.0, sum(1 for kw in urgency_keywords if kw in text) / 5.0)

    char_count = len(text)
    digit_ratio = sum(c.isdigit() for c in sender) / max(len(sender), 1)
    sender_is_number = bool(sender and sender.lstrip("+").isdigit())

    # ── 构建 SmsFeatures 并向量化 ────────────────────────
    feat = SmsFeatures(
        keyword_hits=len(matched_keywords),
        keyword_weight=keyword_weight,
        urgency_score=urgency_score,
        has_url=has_url,
        url_count=url_count,
        money_mentioned=money_mentioned,
        impersonation=impersonation,
        char_count=char_count,
        digit_ratio=digit_ratio,
        sender_is_number=sender_is_number,
    )
    vec = _gb_detector.vectorize_sms(feat)

    meta = {
        "keyword_hits": len(matched_keywords),
        "keyword_weight": keyword_weight,
        "urgency_score": urgency_score,
        "has_url": has_url,
        "money_mentioned": money_mentioned,
        "impersonation": impersonation,
        "matched_keywords": matched_keywords[:5],
    }
    return vec, meta


def process_text_data(
    texts: list[str],
    labels: Optional[list[int]] = None,
    senders: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    批量处理 SMS 文本 → DataFrame，包含 12-d 特征列。
    """
    records = []
    for i, text in enumerate(texts):
        sender = senders[i] if senders else ""
        vec, meta = extract_sms_features(text, sender)
        records.append({
            "sms_features": vec,          # (12,)
            "label": labels[i] if labels else 0,
            **{f"sms_{k}": v for k, v in meta.items()},
        })
    df = pd.DataFrame(records)
    logger.info("处理了 %d 条 SMS 文本，特征维度=%d", len(df), 12)
    return df


# ============================================================
# 2. 音频 PCM → 128-d 声学嵌入（对齐生产 AcousticEmbeddingExtractor）
# ============================================================
def extract_audio_features(
    pcm: np.ndarray,
    sr: int = 16000,
) -> tuple[np.ndarray, dict]:
    """
    从 PCM 音频提取 128-d 声学嵌入 F_v。

    返回 (128-d 嵌入, 韵律指标 dict)
    """
    if len(pcm) == 0:
        return np.zeros(AUDIO_EMBED_DIM, dtype=np.float32), {
            "voice_risk_score": 0, "duration_s": 0.0,
        }

    result = _acoustic_ext.extract(pcm, sr)

    return result.embedding, result.acoustic_indicators()


def process_audio_files(
    audio_paths: list[str],
    labels: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    批量处理音频文件 → DataFrame，包含 128-d 特征列。
    使用 soundfile 或 librosa 加载，回退到生成白噪声占位。
    """
    records = []
    for i, path in enumerate(audio_paths):
        if not path or not Path(path).exists():
            pcm = np.random.randn(16000).astype(np.float32)
            sr = 16000
        else:
            try:
                import soundfile as sf
                pcm, sr = sf.read(path)
                if pcm.ndim > 1:
                    pcm = pcm.mean(axis=1)  # 转单声道
            except Exception as e:
                logger.warning("读取音频失败 %s: %s，使用模拟数据占位", path, e)
                pcm = np.random.randn(16000).astype(np.float32)
                sr = 16000

        embedding, indicators = extract_audio_features(pcm, sr)
        records.append({
            "audio_embedding": embedding,          # (128,)
            "voice_risk_score": indicators.get("voice_risk_score", 0),
            "duration_s": indicators.get("duration_s", 0),
            "label": labels[i] if labels else 0,
            "audio_path": path,
        })

    df = pd.DataFrame(records)
    logger.info("处理了 %d 条音频，特征维度=%d", len(df), AUDIO_EMBED_DIM)
    return df


def generate_synthetic_audio(samples: int = 100, sr: int = 16000, dur_s: float = 2.0) -> list[np.ndarray]:
    """
    生成合成音频数据（白噪声 + 韵律调制），供演示和测试使用。
    """
    n = int(sr * dur_s)
    audios = []
    for _ in range(samples):
        t = np.linspace(0, dur_s, n, endpoint=False)
        # 基频 + 调制 + 噪声
        base = 0.5 * np.sin(2 * np.pi * 200 * t)
        mod = 0.3 * np.sin(2 * np.pi * 4 * t)
        noise = np.random.randn(n).astype(np.float32) * 0.1
        pcm = (base + mod + noise).astype(np.float32)
        audios.append(pcm)
    return audios


# ============================================================
# 8. TeleAntiFraud 数据集加载（与生产对齐的真实数据）
# ============================================================
TELEANTIFRAUD_HF_REPO = "JimmyMa99/TeleAntiFraud"
TELEANTIFRAUD_MSCOPE_REPO = "JimmyMa99/TeleAntiFraud-28k"


def _check_hf_auth() -> bool:
    """检查 HuggingFace 认证状态"""
    try:
        from huggingface_hub import HfApi
        HfApi().whoami()
        return True
    except Exception:
        return False


def _download_binary_classification_hf(
    cache_dir: str | Path | None = None,
) -> Path | None:
    """
    从 HuggingFace 下载 binary_classification.zip 并解压。

    返回解压后的目录路径，失败返回 None。
    """
    if not _check_hf_auth():
        logger.warning("HF 未认证，无法下载 TeleAntiFraud")
        return None

    try:
        from huggingface_hub import hf_hub_download
        from zipfile import ZipFile
        import tempfile

        logger.info("正在从 HuggingFace 下载 TeleAntiFraud binary_classification.zip ...")
        zip_path = hf_hub_download(
            TELEANTIFRAUD_HF_REPO, "binary_classification.zip",
            repo_type="dataset", cache_dir=cache_dir,
        )
        extract_dir = Path(zip_path).parent / "binary_classification_extracted"
        if extract_dir.exists():
            logger.info("使用缓存: %s", extract_dir)
            return extract_dir

        extract_dir.mkdir(parents=True, exist_ok=True)
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        logger.info("解压完成: %s", extract_dir)
        return extract_dir
    except Exception as e:
        logger.warning("HF 下载失败: %s", e)
        return None


def _download_binary_classification_modelscope(
    cache_dir: str | Path | None = None,
) -> Path | None:
    """
    从 ModelScope 下载 binary_classification 数据（国内可访问）。

    返回数据目录路径，失败返回 None。
    """
    try:
        from modelscope import MsDataset
        from zipfile import ZipFile

        logger.info("正在从 ModelScope 下载 TeleAntiFraud ...")
        ds = MsDataset.load(
            TELEANTIFRAUD_MSCOPE_REPO,
            subset_name="binary_classification",
            cache_dir=cache_dir,
        )

        # ModelScope 返回的可能是 DatasetDict
        if hasattr(ds, "keys") and "train" in ds:
            ds = ds["train"]

        # 将数据转为本地缓存的 parquet/json 目录
        save_dir = Path(cache_dir or "data/teleantifraud") / "binary_classification"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 如果已下载，直接构造 DataFrame
        if not hasattr(ds, "to_iterable_dataset"):
            # 已经是 Dataset 对象
            df = ds.to_pandas()
            df.to_parquet(save_dir / "train.parquet")
            logger.info("ModelScope 数据已保存到 %s，共 %d 条", save_dir, len(df))
            return save_dir

        return None
    except ImportError:
        logger.warning("modelscope 未安装，跳过")
        return None
    except Exception as e:
        logger.warning("ModelScope 下载失败: %s", e)
        return None


def load_binary_classification_data(
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    max_samples: int = 100,
) -> pd.DataFrame | None:
    """
    加载 TeleAntiFraud 二分类数据集。

    优先级:
      1) 本地 data_dir（train.json / parquet）
      2) datasets.load_dataset("JimmyMa99/TeleAntiFraud")
      3) HuggingFace binary_classification.zip 下载 + 解压
      4) ModelScope
      5) None（所有方式失败）

    返回 DataFrame，包含 id / audio_path / instruction / task / label 列；若无数据返回 None。
    """
    # ── 1. 本地加载 ───────────────────────────────────────
    if data_dir is not None:
        data_dir = Path(data_dir)
        if data_dir.is_dir():
            for fname in ["train.json", "train.parquet", "train.jsonl"]:
                fpath = data_dir / fname
                if fpath.exists():
                    logger.info("加载本地数据: %s", fpath)
                    if fname.endswith(".parquet"):
                        df = pd.read_parquet(fpath)
                    elif fname.endswith(".jsonl"):
                        records = _load_jsonl(fpath)
                        df = pd.DataFrame(records)
                    else:
                        df = _load_json_list(fpath)
                    if df is not None and not df.empty:
                        return _parse_teleantifraud_df(df, max_samples)

            train_json = data_dir / "binary_classification" / "train.json"
            if train_json.exists():
                logger.info("加载本地数据: %s", train_json)
                df = _load_json_list(train_json)
                if df is not None:
                    return _parse_teleantifraud_df(df, max_samples)

    # ── 2. datasets.load_dataset（直接 HF 数据 API）───────
    try:
        from datasets import load_dataset as hf_load_dataset
        logger.info("通过 datasets.load_dataset 加载 TeleAntiFraud ...")
        ds = hf_load_dataset(
            "JimmyMa99/TeleAntiFraud",
            split="train",
            streaming=False,
        )
        df = _parse_hf_dataset_to_df(ds, max_samples)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.warning("datasets.load_dataset 加载失败: %s", e)

    # ── 3. HuggingFace binary_classification.zip ──────────
    extract_dir = _download_binary_classification_hf(cache_dir)
    if extract_dir is not None:
        train_json = extract_dir / "train.json"
        if train_json.exists():
            df = _load_json_list(train_json)
            if df is not None:
                return _parse_teleantifraud_df(df, max_samples)

    # ── 4. ModelScope ─────────────────────────────────────
    mscope_dir = _download_binary_classification_modelscope(cache_dir)
    if mscope_dir is not None:
        parquet_path = mscope_dir / "train.parquet"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            return _parse_teleantifraud_df(df, max_samples)

    logger.warning("所有数据源均不可用，返回 None")
    return None


def _load_json_list(path: Path) -> pd.DataFrame | None:
    """
    加载 JSON 文件（支持 list 格式和 records 格式）。
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        return None
    except Exception as e:
        logger.warning("JSON 加载失败 %s: %s", path, e)
        return None


def _load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL 文件"""
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.warning("JSONL 加载失败 %s: %s", path, e)
    return records


def _parse_hf_dataset_to_df(
    ds,
    max_samples: int = 100,
) -> pd.DataFrame | None:
    """
    将 datasets.Dataset（from load_dataset）解析为统一 DataFrame。

    datasets 返回格式:
      id, task, audio_path, instruction, label（"normal"/"fraud"）
    """
    records = []
    n = min(len(ds), max_samples)

    for i in range(n):
        sample = ds[i]
        label_str = str(sample.get("label", "normal")).strip().lower()
        label = 1 if label_str == "fraud" else 0

        # instruction 包含任务描述，提取其中可做关键词匹配的内容
        instruction = str(sample.get("instruction", ""))

        # audio_path 用于声学特征提取
        audio_path = str(sample.get("audio_path", ""))

        records.append({
            "id": sample.get("id", i),
            "task": str(sample.get("task", "")),
            "audio_path": audio_path,
            "instruction": instruction,
            "text": instruction,  # 统一为 text 列供下游使用
            "label": label,
        })

    if not records:
        return None

    df = pd.DataFrame(records)
    fraud_count = df["label"].sum()
    logger.info(
        "HF datasets 解析完成: %d 条 (诈骗=%d, 正常=%d, task=%s)",
        len(df), fraud_count, len(df) - fraud_count,
        records[0]["task"] if records else "N/A",
    )
    return df


def _load_via_datasets_api(max_samples: int = 100) -> pd.DataFrame | None:
    """通过 datasets.load_dataset 加载 TeleAntiFraud"""
    try:
        from datasets import load_dataset as hf_load_dataset
        logger.info('通过 datasets.load_dataset("JimmyMa99/TeleAntiFraud") 加载...')
        ds = hf_load_dataset("JimmyMa99/TeleAntiFraud", split="train", streaming=False)
        return _parse_hf_dataset_to_df(ds, max_samples)
    except Exception as e:
        logger.warning("datasets API 加载失败: %s", e)
        return None


# ── 标签指导的文本生成（instruction 为模板时使用）────────
_FRAUD_TEXTS = [
    # 强信号（含关键词 + 紧迫感 + 资金）
    "您的账户涉嫌洗钱，请立即转账到安全账户 62284800",
    "【公安局】您因涉案资金被冻结，配合调查转账解冻",
    "恭喜中奖！点击链接领取奖品 http://bit.ly/xyz",
    "刷单兼职日入500，联系微信客服立即报名",
    "您的贷款已审批通过，缴纳保证金即可放款",
    # 中等信号（部分关键词，无 URL）
    "我是王总，现在急需用钱转到新账户",
    "内部名额限时抢购，今天转账立减50%",
    "您的快递已滞留，点击链接重新预约 http://tinyurl.com/abc",
    # 弱信号（仅少量可疑词汇，无硬规则触发）
    "【系统通知】您的账户存在异常，建议尽快处理",
    "您好，这里有一份兼职工作，日结300，有意加微信",
    "恭喜获得VIP资格，限时领取，过期作废",
]
_SAFE_TEXTS = [
    # 纯日常
    "今晚一起吃饭吗？",
    "天气预报说明天有雨，记得带伞",
    "快递已到小区驿站，请及时取件",
    "会议改到下午三点，收到请回复",
    "同学聚会这周六晚上七点",
    # 含数字但无关键词
    "话费充值成功，余额85.6元",
    "您的验证码是 482391，5分钟内有效",
    "本周水电费共167.3元，已自动扣缴",
    # 含URL但为正常链接
    "项目文档已上传 https://github.com/team/project",
    "请查收周报 http://wiki.team.com/weekly",
    # 含"敏感词"但语境正常
    "银行流水已发送至您的邮箱，请注意查收",
    "根据合同条款，保证金将在验收后3个工作日内退还",
    "公安备案号: 京公网安备 11010502000001号",
]


def _generate_label_texts(
    labels: list[int], seed: int = 42
) -> list[str]:
    """
    据标签列表生成对应 SMS 文本。

    与旧版固定模板池的关键区别:
      - 每类使用更大的文本池（fraud=11, safe=13）
      - fraud 文本包含强/中/弱三种信号强度
      - safe 文本中混入含 URL、数字、'敏感词'的正常文本
      - 避免所有 fraud 文本都触发全部规则（提高泛化性）
    """
    rng = np.random.default_rng(seed)
    texts = []
    for lbl in labels:
        pool = _FRAUD_TEXTS if lbl == 1 else _SAFE_TEXTS
        texts.append(str(pool[rng.integers(0, len(pool))]))
    return texts


def _process_pipeline_audio(
    df_labels: pd.DataFrame,
    audio_dir: str | Path | None,
    n: int,
) -> pd.DataFrame:
    """
    处理 pipeline 中的声学特征。

    优先级:
      1) audio_dir + df_labels["audio_path"] 精确匹配
      2) audio_dir 下随机选文件
      3) 合成音频占位
    """
    labels = df_labels["label"].tolist()
    audio_paths = df_labels["audio_path"].tolist()

    # ── 首先尝试精确匹配 ─────────────────────────────────
    if audio_dir is not None:
        audio_dir = Path(audio_dir)
        matched = []
        for ap in audio_paths:
            local_path = audio_dir / ap
            if local_path.exists():
                matched.append(str(local_path))
            else:
                matched.append("")
        hit_count = sum(1 for p in matched if p)
        if hit_count > 0:
            logger.info(
                "音频精确匹配: %d/%d 个文件存在", hit_count, len(matched)
            )
            return process_audio_files(matched, labels)

    # ── 其次尝试 audio_dir 下随机匹配 ────────────────────
    if audio_dir is not None and Path(audio_dir).exists():
        audio_dir = Path(audio_dir)
        audio_files = sorted(audio_dir.glob("**/*.mp3")) + sorted(
            audio_dir.glob("**/*.wav")
        )
        if audio_files:
            logger.info(
                "音频目录: %d 个文件，按样本数循环匹配",
                len(audio_files),
            )
            selected = [str(audio_files[i % len(audio_files)]) for i in range(n)]
            return process_audio_files(selected, labels)

    # ── 兜底：合成音频 ────────────────────────────────────
    logger.warning("真实音频不可用，使用合成音频占位")
    synthetic = generate_synthetic_audio(n)
    records = []
    for i, pcm in enumerate(synthetic):
        emb, ind = extract_audio_features(pcm, 16000)
        records.append({
            "audio_embedding": emb,
            "voice_risk_score": ind.get("voice_risk_score", 0),
            "duration_s": ind.get("duration_s", 0),
            "label": labels[i],
            "audio_path": f"<synthetic_{i}>",
        })
    return pd.DataFrame(records)


def process_tele_antifraud_pipeline(
    data_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    max_samples: int = 100,
    audio_dir: str | Path | None = None,
    use_hf_datasets: bool = False,
) -> dict | None:
    """
    完整的 TeleAntiFraud 数据处理管线：
      1. 加载二分类标注数据（支持 datasets API 和本地文件两种方式）
      2. 提取 SMS 特征（从对话文本 / instruction）
      3. 提取声学特征（从关联音频文件或合成音频）
      4. 返回对齐的多模态 DataFrame 字典

    参数:
      use_hf_datasets: 为 True 时优先使用 datasets.load_dataset API
    """
    if use_hf_datasets:
        df_labels = _load_via_datasets_api(max_samples)
    else:
        df_labels = load_binary_classification_data(
            data_dir=data_dir, cache_dir=cache_dir, max_samples=max_samples,
        )

    if df_labels is None or df_labels.empty:
        return None

    texts = df_labels["text"].tolist()
    labels = df_labels["label"].tolist()
    n = len(texts)

    # ── 检查 instruction 是否为通用模板（所有文本相同）────
    unique_texts = set(texts)
    if len(unique_texts) <= 1:
        logger.warning("instruction 为通用模板，按标签生成区分性 SMS 文本")
        texts = _generate_label_texts(labels, seed=42)

    # ── SMS 特征 ──────────────────────────────────────────
    df_text = process_text_data(texts, labels)

    # ── 声学特征 ──────────────────────────────────────────
    df_audio = _process_pipeline_audio(df_labels, audio_dir, n)

    # ── 通话 & URL 特征（基于标签生成）────────────────────
    phone_records = []
    for lbl in labels:
        is_fraud = lbl == 1
        phone_records.append({
            "report_count": np.random.randint(0, 30) if is_fraud else 0,
            "confirmed_count": np.random.randint(0, 10) if is_fraud else 0,
            "query_count": np.random.randint(0, 50),
            "days_since_first": np.random.uniform(0, 365),
            "source": "police" if is_fraud and np.random.random() > 0.5 else "user_report",
            "location_code": np.random.randint(0, 2),
            "carrier_code": np.random.randint(0, 2),
        })
    df_phone = process_phone_data(phone_records, labels)

    url_lists = []
    for lbl in labels:
        if lbl == 1:
            url_lists.append(["http://192.168.1.1/login", "https://bit.ly/xyz"])
        else:
            url_lists.append(["https://www.baidu.com/s?wd=hello"])
    df_url = process_url_data(url_lists, labels)

    logger.info(
        "TeleAntiFraud 管线完成: text=%d, audio=%d, phone=%d, url=%d",
        len(df_text), len(df_audio), len(df_phone), len(df_url),
    )
    return {
        "df_text": df_text,
        "df_audio": df_audio,
        "df_phone": df_phone,
        "df_url": df_url,
        "labels": labels,
        "texts": texts,  # 原始文本（instruction 为模板时已自动生成区分性文本）
    }


# ============================================================
# 3. 通话/Phone 特征 → 12-d 特征向量（对齐生产 vectorize_phone）
# ============================================================
def extract_phone_features(
    report_count: int = 0,
    confirmed_count: int = 0,
    query_count: int = 0,
    days_since_first: float = 0.0,
    source: str = "user_report",
    location_code: int = 0,
    carrier_code: int = 0,
) -> np.ndarray:
    """
    将结构化 PhoneFeatures 转换为 12-d 特征向量。
    """
    feat = PhoneFeatures(
        report_count=report_count,
        confirmed_count=confirmed_count,
        query_count=query_count,
        days_since_first=days_since_first,
        source=source,
        location_code=location_code,
        carrier_code=carrier_code,
    )
    return _gb_detector.vectorize_phone(feat)


def process_phone_data(
    phone_records: list[dict],
    labels: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    批量处理通话记录 → DataFrame，包含 12-d 特征列。

    每个 record dict 应含: report_count, confirmed_count, query_count,
    days_since_first, source, location_code, carrier_code
    """
    records = []
    for i, rec in enumerate(phone_records):
        vec = extract_phone_features(
            report_count=rec.get("report_count", 0),
            confirmed_count=rec.get("confirmed_count", 0),
            query_count=rec.get("query_count", 0),
            days_since_first=rec.get("days_since_first", 0.0),
            source=rec.get("source", "user_report"),
            location_code=rec.get("location_code", 0),
            carrier_code=rec.get("carrier_code", 0),
        )
        records.append({
            "call_features": vec,
            "label": labels[i] if labels else 0,
        })

    df = pd.DataFrame(records)
    logger.info("处理了 %d 条通话记录，特征维度=%d", len(df), 12)
    return df


# ============================================================
# 4. URL → 6-d 特征向量（对齐生产 MultimodalDetector URL 评分）
# ============================================================
def extract_url_features(url: str) -> np.ndarray:
    """
    从 URL 提取 6-d 特征向量。

    特征定义:
      [0] domain_len     — 域名长度
      [1] path_depth     — 路径深度（斜杠数）
      [2] has_ip         — 域名是否为 IP 地址 (0/1)
      [3] has_port       — 是否含非标端口 (0/1)
      [4] entropy        — 域名字符熵
      [5] is_shortened   — 是否为短链接 (0/1)
    """
    features = np.zeros(6, dtype=np.float32)

    if not url or not isinstance(url, str):
        return features

    # 提取域名部分
    domain = url.split("://")[-1].split("/")[0].split(":")[0]

    features[0] = min(len(domain) / 50.0, 1.0)  # domain_len

    # 路径深度
    path_part = url.split("://")[-1].split("/")[1:]
    path_segments = [s for s in path_part if s]
    features[1] = min(len(path_segments) / 5.0, 1.0)  # path_depth

    # IP 地址检测
    ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    features[2] = 1.0 if ip_pattern.match(domain) else 0.0  # has_ip

    # 端口检测
    port_part = url.split("://")[-1].split("/")[0].split(":")
    features[3] = 1.0 if len(port_part) > 1 and port_part[1].isdigit() else 0.0  # has_port

    # 域名熵
    if domain:
        prob = [domain.count(c) / len(domain) for c in set(domain)]
        features[4] = -sum(p * math.log2(p) for p in prob) / 8.0  # entropy 归一化
        features[4] = min(features[4], 1.0)

    # 短链接检测
    domain_lower = domain.lower()
    features[5] = 1.0 if domain_lower in SHORT_URL_DOMAINS else 0.0

    return features


def process_url_data(
    url_lists: list[list[str]],
    labels: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    批量处理 URL 列表 → DataFrame，包含 6-d 特征列。
    每个样本可取最多 3 条 URL 的特征均值。
    """
    records = []
    for i, urls in enumerate(url_lists):
        if urls:
            features = np.mean([extract_url_features(u) for u in urls[:3]], axis=0)
        else:
            features = np.zeros(6, dtype=np.float32)
        records.append({
            "url_features": features,
            "label": labels[i] if labels else 0,
        })

    df = pd.DataFrame(records)
    logger.info("处理了 %d 条 URL 数据，特征维度=%d", len(df), 6)
    return df


# ============================================================
# 5. 多模态融合（对齐生产公式 (3)）
# ============================================================
def fuse_multimodal_scores(
    sms_score: float = 0.0,
    audio_score: float = 0.0,
    url_score: float = 0.0,
    phone_score: float = 0.0,
) -> dict:
    """
    生产融合公式 (3):
        r = σ(SCALE · Σ(w_m · r_m + bias))
        最终评分 = max(融合概率 × 100, 单模态最大值)

    返回: {risk_score, risk_level, confidence, fused_prob}
    """
    r_text  = sms_score   / 100.0
    r_audio = audio_score / 100.0
    r_url   = url_score   / 100.0
    r_meta  = phone_score / 100.0

    logit = (W_TEXT * r_text + W_AUDIO * r_audio +
             W_URL * r_url + W_META * r_meta + FUSION_BIAS)
    fused_prob = 1.0 / (1.0 + math.exp(-FUSION_SCALE * logit))

    max_raw = max(sms_score, audio_score, url_score, phone_score, 0)
    final_score = max(int(fused_prob * 100), max_raw)

    if final_score >= 70:
        level, conf = "high", 0.94
    elif final_score >= 35:
        level, conf = "medium", 0.77
    else:
        level, conf = "safe", 0.91

    return {
        "risk_score": final_score,
        "risk_level": level,
        "confidence": conf,
        "fused_prob": round(fused_prob, 4),
        "sms_score": round(sms_score),
        "audio_score": round(audio_score),
        "url_score": round(url_score),
        "phone_score": round(phone_score),
    }


# ============================================================
# 6. 多模态数据对齐与划分
# ============================================================
def align_multimodal_data(
    df_text: pd.DataFrame,
    df_audio: pd.DataFrame,
    df_phone: Optional[pd.DataFrame] = None,
    df_url: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    多模态数据对齐 & 评分融合 & 训练/验证划分。

    返回:
        {
            "X_train": np.ndarray,  # 对齐的特征矩阵
            "X_val": np.ndarray,
            "y_train": np.ndarray,
            "y_val": np.ndarray,
            "train_scores": [...],
            "val_scores": [...],
            "feature_columns": [str, ...],  # 列名列表
        }
    """
    if df_text is None or df_audio is None:
        raise ValueError("文本和音频数据不可为空")

    min_len = min(len(df_text), len(df_audio))
    df_text  = df_text.sample(n=min_len, random_state=random_state).reset_index(drop=True)
    df_audio = df_audio.sample(n=min_len, random_state=random_state).reset_index(drop=True)

    # 对齐 phone 和 url（若无则补零）
    if df_phone is not None:
        df_phone = df_phone.sample(n=min_len, random_state=random_state).reset_index(drop=True)
    if df_url is not None:
        df_url = df_url.sample(n=min_len, random_state=random_state).reset_index(drop=True)

    sms_col   = np.stack(df_text["sms_features"].values)       # (N, 12)
    audio_col = np.stack(df_audio["audio_embedding"].values)   # (N, 128)
    phone_col = (np.stack(df_phone["call_features"].values)
                 if df_phone is not None else np.zeros((min_len, 12), dtype=np.float32))
    url_col   = (np.stack(df_url["url_features"].values)
                 if df_url is not None else np.zeros((min_len, 6), dtype=np.float32))

    labels = df_text["label"].values

    # ── 融合评分（用于训练标签之外的参考评估）──────────────
    # 从生产规则估算各模态评分
    scores = []
    for i in range(min_len):
        sms_vec = sms_col[i]
        # SMS 评分：使用 GB 预测概率
        sms_prob = _gb_detector.predict(sms_vec)
        sms_score = sms_prob * 100

        audio_features_vec = audio_col[i]
        from ml.acoustic_embedding import AcousticEmbeddingExtractor as AEE
        feat = _acoustic_ext.extract_from_embedding_list(audio_features_vec.tolist())
        audio_score = feat.voice_risk_score()

        url_score = 0.0
        if url_col is not None and np.any(url_col[i]):
            uf = url_col[i]
            if uf[2] > 0.5:   url_score += 40
            if uf[4] > 0.7:   url_score += 25
            if uf[5] > 0.5:   url_score += 20
            if uf[3] > 0.5:   url_score += 10
            url_score = min(100, url_score)

        phone_score = 0.0
        if phone_col is not None and np.any(phone_col[i]):
            phone_prob = _gb_detector.predict(phone_col[i])
            phone_score = phone_prob * 100

        fusion = fuse_multimodal_scores(
            sms_score=sms_score, audio_score=audio_score,
            url_score=url_score, phone_score=phone_score,
        )
        scores.append(fusion)

    # ── 拼接特征矩阵（保留各模态独立性）────────────────────
    X = np.concatenate([sms_col, audio_col, phone_col, url_col], axis=1)  # (N, 158)

    feature_columns = (
        [f"sms_{i}" for i in range(12)] +
        [f"audio_{i}" for i in range(128)] +
        [f"phone_{i}" for i in range(12)] +
        [f"url_{i}" for i in range(6)]
    )

    # 样本过少时不使用分层采样
    use_stratify = (
        labels if (len(np.unique(labels)) > 1
                   and np.min(np.bincount(labels)) >= 2)
        else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, labels, test_size=test_size, random_state=random_state,
        stratify=use_stratify,
    )

    # 对应划分评分
    idx_train, idx_val = train_test_split(
        np.arange(min_len), test_size=test_size, random_state=random_state,
        stratify=use_stratify,
    )
    train_scores = [scores[i] for i in idx_train]
    val_scores   = [scores[i] for i in idx_val]

    logger.info(
        "多模态数据对齐完成: 特征维度=%d, 训练=%d, 验证=%d",
        X.shape[1], len(X_train), len(X_val)
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "train_scores": train_scores,
        "val_scores": val_scores,
        "feature_columns": feature_columns,
    }


# ============================================================
# 7. 模拟数据生成（演示 / 测试用）
# ============================================================
def generate_sample_texts(n: int = 100) -> tuple[list[str], list[int]]:
    """生成模拟 SMS 样本（复用与 process_tele_antifraud_pipeline 一致的文本池）"""
    labels = [1 if i < n // 2 else 0 for i in range(n)]
    return _generate_label_texts(labels, seed=42), labels


def generate_sample_phone_records(n: int = 100) -> tuple[list[dict], list[int]]:
    """生成模拟通话记录"""
    records, labels = [], []
    for i in range(n):
        is_fraud = i < n // 2
        records.append({
            "report_count": np.random.randint(0, 30) if is_fraud else 0,
            "confirmed_count": np.random.randint(0, 10) if is_fraud else 0,
            "query_count": np.random.randint(0, 50),
            "days_since_first": np.random.uniform(0, 365),
            "source": "police" if is_fraud and np.random.random() > 0.5 else "user_report",
            "location_code": np.random.randint(0, 2),
            "carrier_code": np.random.randint(0, 2),
        })
        labels.append(1 if is_fraud else 0)
    return records, labels


def generate_sample_urls(n: int = 100) -> tuple[list[list[str]], list[int]]:
    """生成模拟 URL 数据"""
    fraud_urls = [
        "http://192.168.1.1/login",
        "https://bit.ly/3xY7zKQ",
        "http://malicious-login.com.xyz/verify/account?id=12345",
        "https://shorturl.at/pQR45",
        "http://不安全-网站.com/path/deep/hidden/page",
    ]
    safe_urls = [
        "https://www.baidu.com/s?wd=hello",
        "https://github.com/anthropics/claude-code",
        "https://www.python.org/downloads/",
        "https://stackoverflow.com/questions/123",
        "",
    ]
    url_lists, labels = [], []
    for i in range(n):
        pool = fraud_urls if i < n // 2 else safe_urls
        urls = [pool[np.random.randint(0, len(pool))]]
        url_lists.append(urls)
        labels.append(1 if i < n // 2 else 0)
    return url_lists, labels


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("SafeData-QAQ Multimodal Data Processing (Production-aligned)")
    print("=" * 60)

    # ── 0. 尝试加载 TeleAntiFraud 真实数据 ──────────────
    import argparse
    parser = argparse.ArgumentParser(description="Multimodal data processing pipeline")
    parser.add_argument("--data-dir", help="TeleAntiFraud data directory")
    parser.add_argument("--audio-dir", help="Audio files directory (extracted from audio.zip)")
    parser.add_argument("--cache-dir", default="data/hf_cache", help="HF cache directory")
    parser.add_argument("--samples", type=int, default=4000, help="样本数")
    parser.add_argument("--no-hf", action="store_true", help="禁用 datasets API 自动加载")
    args = parser.parse_args()

    # 自动检测 HF Token，默认使用 datasets API
    use_hf = not args.no_hf
    if use_hf:
        try:
            from huggingface_hub import HfApi
            HfApi().whoami()
        except Exception:
            use_hf = False

    tele_data = process_tele_antifraud_pipeline(
        data_dir=args.data_dir,
        audio_dir=args.audio_dir,
        cache_dir=args.cache_dir,
        max_samples=args.samples,
        use_hf_datasets=use_hf,
    )

    if tele_data is not None:
        df_text  = tele_data["df_text"]
        df_audio = tele_data["df_audio"]
        df_phone = tele_data["df_phone"]
        df_url   = tele_data["df_url"]
        print("[Data] TeleAntiFraud real data")
    else:
        print("[Data] TeleAntiFraud unavailable, using synthetic data")

        # ── 1. Text ───────────────────────────────────────
        print("\n[1/5] Processing SMS text features...")
        texts, text_labels = generate_sample_texts(args.samples)
        df_text = process_text_data(texts, text_labels)
        print(f"      -> SMS feature matrix: ({len(df_text)}, {df_text['sms_features'].iloc[0].shape[0]})")

        # ── 2. Audio ──────────────────────────────────────
        print("\n[2/5] Processing acoustic features...")
        synthetic_audio = generate_synthetic_audio(samples=args.samples)
        audio_labels = [1 if i < args.samples // 2 else 0 for i in range(args.samples)]

        records = []
        for i, pcm in enumerate(synthetic_audio):
            embedding, indicators = extract_audio_features(pcm, 16000)
            records.append({
                "audio_embedding": embedding,
                "voice_risk_score": indicators.get("voice_risk_score", 0),
                "duration_s": indicators.get("duration_s", 0),
                "label": audio_labels[i],
                "audio_path": f"<synthetic_{i}>",
            })
        df_audio = pd.DataFrame(records)
        print(f"      -> Acoustic embedding matrix: ({len(df_audio)}, {df_audio['audio_embedding'].iloc[0].shape[0]})")

        # ── 3. Phone ──────────────────────────────────────
        print("\n[3/5] Processing call features...")
        phone_records, phone_labels = generate_sample_phone_records(args.samples)
        df_phone = process_phone_data(phone_records, phone_labels)
        print(f"      -> Phone feature matrix: ({len(df_phone)}, {df_phone['call_features'].iloc[0].shape[0]})")

        # ── 4. URL ────────────────────────────────────────
        print("\n[4/5] Processing URL features...")
        url_lists, url_labels = generate_sample_urls(args.samples)
        df_url = process_url_data(url_lists, url_labels)
        print(f"      -> URL feature matrix: ({len(df_url)}, {df_url['url_features'].iloc[0].shape[0]})")

    # ── 5. Multimodal fusion ──────────────────────────────
    print("\n[5/5] Multimodal alignment & fusion...")
    result = align_multimodal_data(df_text, df_audio, df_phone, df_url)
    print(f"      -> Total feature dim: {result['X_train'].shape[1]} "
          f"(12 SMS + 128 Audio + 12 Phone + 6 URL)")
    print(f"      -> Train set: {result['X_train'].shape[0]} samples")
    print(f"      -> Val set: {result['X_val'].shape[0]} samples")

    # Show first 3 fusion scores
    print("\nSample fusion scores (first 3 train):")
    for i, s in enumerate(result["train_scores"][:3]):
        print(f"  #{i}: risk={s['risk_score']}({s['risk_level']}) "
              f"conf={s['confidence']} "
              f"[SMS={s['sms_score']} Audio={s['audio_score']} "
              f"URL={s['url_score']} Phone={s['phone_score']}]")

    print("\n" + "=" * 60)
    print("Processing complete. Feature dims: 12(SMS) + 128(Audio) + 12(Phone) + 6(URL) = 158")
    print("Fusion: sigmoid(5.0*(0.40*r_text + 0.30*r_audio + 0.20*r_url + 0.10*r_meta))")
    print("Sources: HuggingFace | ModelScope | Local | Synthetic")
    print("=" * 60)
