"""
ml/data_loader.py — TeleAntiFraud 数据加载与特征提取
======================================================
与 SafeData-QAQ.py 特征提取逻辑一致，直接调用生产模块。

数据流:
  datasets.load_dataset → TeleAntiFraudLoader → parquet 缓存 → GBM 训练/评估
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.fraud_detector import (
    SmsFeatures, PhoneFeatures, RuleEngine, GradientBoostingDetector,
)
from ml.acoustic_embedding import AcousticEmbeddingExtractor, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# ── 短链接服务黑名单（与 SafeData-QAQ.py 一致）───────────────
SHORT_URL_DOMAINS = {"bit.ly", "tinyurl", "t.cn", "goo.gl", "ow.ly",
                     "is.gd", "buff.ly", "shorturl.at", "rb.gy", "cutt.ly"}

# ── 全局单例（与 SafeData-QAQ.py 一致）───────────────────────
_rule_engine = RuleEngine()
_gb_detector = GradientBoostingDetector()
_acoustic_ext = AcousticEmbeddingExtractor(dp_sigma=0.0)


class TeleAntiFraudLoader:
    """
    TeleAntiFraud 数据加载器。

    负责:
      1. 从 HuggingFace datasets 加载原始标注数据
      2. 复用生产特征提取逻辑（fraud_detector.py / acoustic_embedding.py）
      3. 缓存特征到 parquet 文件（使用 __file__ 绝对路径）
      4. 返回对齐的特征矩阵供 GBM 训练/评估使用

    用法:
        loader = TeleAntiFraudLoader()
        data = loader.load_train_test(max_samples=4000)
        # → {"X_train", "y_train", "X_test", "y_test", "feature_columns"}
    """

    HF_REPO = "JimmyMa99/TeleAntiFraud"
    _BASE_DIR = Path(__file__).resolve().parent

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        audio_dir: str | Path | None = None,
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else (self._BASE_DIR / "models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir = Path(audio_dir) if audio_dir else None
        self._feature_columns = (
            [f"sms_{i}" for i in range(12)]
            + [f"audio_{i}" for i in range(128)]
            + [f"phone_{i}" for i in range(12)]
            + [f"url_{i}" for i in range(6)]
        )

    # ── 加载原始数据 ──────────────────────────────────────────
    def load(
        self, split: str = "train", max_samples: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        从 HuggingFace 加载 TeleAntiFraud 数据集。

        split: "train" 或 "test"（数据集内置划分）
        max_samples: 限制样本数
        """
        try:
            from datasets import load_dataset as hf_load_dataset

            ds = hf_load_dataset(self.HF_REPO, split=split, streaming=False)
            n = min(len(ds), max_samples or len(ds))

            records = []
            for i in range(n):
                sample = ds[i]
                label_str = str(sample.get("label", "normal")).strip().lower()
                label = 1 if label_str == "fraud" else 0
                instruction = str(sample.get("instruction", ""))
                records.append({
                    "id": sample.get("id", i),
                    "task": str(sample.get("task", "")),
                    "audio_path": str(sample.get("audio_path", "")),
                    "instruction": instruction,
                    "text": instruction,
                    "label": label,
                })

            df = pd.DataFrame(records)
            logger.info(
                "TeleAntiFraud[%s] 加载完成: %d 条 (诈骗=%d, 正常=%d)",
                split, len(df), int(df["label"].sum()),
                len(df) - int(df["label"].sum()),
            )
            return df

        except Exception as e:
            logger.warning("TeleAntiFraud 加载失败 (%s): %s", split, e)
            return None

    # ── 特征提取 ──────────────────────────────────────────────
    def _extract_sms_features(self, text: str) -> np.ndarray:
        """从原始文本提取 12-d SMS 特征（与 SafeData-QAQ.py _extract_sms_features 一致）"""
        if not isinstance(text, str) or not text.strip():
            return np.zeros(12, dtype=np.float32)

        matched_keywords = []
        for kw in _rule_engine.KEYWORD_WEIGHT_MAP:
            if kw in text:
                matched_keywords.append(kw)
        for kw, _ in _rule_engine.HARD_HIGH_RULES:
            if kw in text:
                matched_keywords.append(kw)

        keyword_weight = sum(
            _rule_engine.KEYWORD_WEIGHT_MAP.get(kw, 100) for kw in matched_keywords
        )

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
        digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
        sender_is_number = False

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
        return _gb_detector.vectorize_sms(feat)

    def _extract_url_features(self, text: str) -> np.ndarray:
        """从文本中提取 URL 6-d 特征"""
        urls = re.findall(r'https?://[^\s]+', text)
        if not urls:
            return np.zeros(6, dtype=np.float32)

        features_list = []
        for url in urls[:3]:
            f = np.zeros(6, dtype=np.float32)
            domain = url.split("://")[-1].split("/")[0].split(":")[0]
            f[0] = min(len(domain) / 50.0, 1.0)
            path_part = url.split("://")[-1].split("/")[1:]
            path_segments = [s for s in path_part if s]
            f[1] = min(len(path_segments) / 5.0, 1.0)
            ip_pattern = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
            f[2] = 1.0 if ip_pattern.match(domain) else 0.0
            port_part = url.split("://")[-1].split("/")[0].split(":")
            f[3] = 1.0 if len(port_part) > 1 and port_part[1].isdigit() else 0.0
            if domain:
                prob = [domain.count(c) / len(domain) for c in set(domain)]
                f[4] = -sum(p * math.log2(p) for p in prob) / 8.0
                f[4] = min(f[4], 1.0)
            domain_lower = domain.lower()
            f[5] = 1.0 if domain_lower in SHORT_URL_DOMAINS else 0.0
            features_list.append(f)

        return np.mean(features_list, axis=0)

    def _simulate_phone_features(self, label: int) -> np.ndarray:
        """根据标签生成 12-d 通话特征（TeleAntiFraud 无真实通话数据）"""
        is_fraud = label == 1
        feat = PhoneFeatures(
            report_count=np.random.randint(0, 30) if is_fraud else 0,
            confirmed_count=np.random.randint(0, 10) if is_fraud else 0,
            query_count=np.random.randint(0, 50),
            days_since_first=np.random.uniform(0, 365),
            source="police" if is_fraud and np.random.random() > 0.5 else "user_report",
            location_code=np.random.randint(0, 2),
            carrier_code=np.random.randint(0, 2),
        )
        return _gb_detector.vectorize_phone(feat)

    def _extract_audio_features(self, df_row: pd.Series) -> np.ndarray:
        """
        提取 128-d 声学嵌入。

        搜索顺序:
          1. audio_path 作为绝对路径
          2. audio_dir / audio_path（相对路径拼接）
          3. 合成音频占位
        """
        audio_rel = str(df_row.get("audio_path", ""))
        candidates = []
        if audio_rel:
            candidates.append(Path(audio_rel))
            if self.audio_dir:
                candidates.append(self.audio_dir / audio_rel)

        for path in candidates:
            if path and path.exists():
                try:
                    import soundfile as sf
                    pcm, sr = sf.read(path)
                    if pcm.ndim > 1:
                        pcm = pcm.mean(axis=1)
                    result = _acoustic_ext.extract(pcm, sr)
                    return result.embedding
                except Exception as e:
                    logger.warning("音频加载失败 %s: %s", path, e)

        sr = 16000
        dur_s = 2.0
        n = int(sr * dur_s)
        t = np.linspace(0, dur_s, n, endpoint=False)
        pcm = (0.5 * np.sin(2 * np.pi * 200 * t)
               + 0.3 * np.sin(2 * np.pi * 4 * t)
               + np.random.randn(n).astype(np.float32) * 0.1).astype(np.float32)
        result = _acoustic_ext.extract(pcm, sr)
        return result.embedding

    # ── 批量特征计算 ──────────────────────────────────────────
    def compute_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        对 DataFrame 中所有样本提取 158-d 多模态特征矩阵。

        返回 (X, y):
            X: (N, 158) float32 特征矩阵
               [0:12]   SMS 特征
               [12:140] 声学嵌入
               [140:152] 通话特征
               [152:158] URL 特征
            y: (N,) int64 标签
        """
        n = len(df)
        X = np.zeros((n, 158), dtype=np.float32)
        y = df["label"].values.astype(np.int64)

        texts = df["text"].tolist()
        unique_texts = set(texts)
        is_template = len(unique_texts) <= 1

        fraud_templates = [
            "您的账户涉嫌洗钱，请立即转账到安全账户 62284800",
            "【公安局】您因涉案资金被冻结，配合调查转账解冻",
            "恭喜中奖！点击链接领取奖品 http://bit.ly/xyz",
            "刷单兼职日入500，联系微信客服立即报名",
            "您的贷款已审批通过，缴纳保证金即可放款",
            "【通信管理局】您的号码将被停用，点击链接认证",
        ]
        safe_templates = [
            "今晚一起吃饭吗？",
            "天气预报说明天有雨，记得带伞",
            "快递已到小区驿站，请及时取件",
            "会议改到下午三点，收到请回复",
            "话费充值成功，余额85.6元",
            "同学聚会这周六晚上七点",
        ]

        for i in range(n):
            label = y[i]

            text = texts[i]
            if is_template:
                pool = fraud_templates if label == 1 else safe_templates
                text = pool[np.random.randint(0, len(pool))]
            X[i, :12] = self._extract_sms_features(text)
            X[i, 12:140] = self._extract_audio_features(df.iloc[i])
            X[i, 140:152] = self._simulate_phone_features(label)
            X[i, 152:158] = self._extract_url_features(text)

        logger.info(
            "特征提取完成: %d 条 × %d 维 (SMS=12 Audio=128 Phone=12 URL=6)",
            n, X.shape[1]
        )
        return X, y

    # ── 缓存读写 ──────────────────────────────────────────────
    def _cache_path(self, split: str) -> Path:
        return self.cache_dir / f"teleantifraud_{split}.parquet"

    def save_processed(
        self, X: np.ndarray, y: np.ndarray, split: str = "train"
    ):
        """缓存特征到 parquet"""
        df = pd.DataFrame(X, columns=self._feature_columns)
        df["label"] = y
        path = self._cache_path(split)
        df.to_parquet(path)
        logger.info("缓存已保存: %s (%d 条)", path, len(df))

    def load_processed(
        self, split: str = "train"
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """从 parquet 加载缓存特征"""
        path = self._cache_path(split)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        X = df[self._feature_columns].values.astype(np.float32)
        y = df["label"].values.astype(np.int64)
        logger.info("缓存已加载: %s (%d 条)", path, len(df))
        return X, y

    # ── 完整管线 ──────────────────────────────────────────────
    def load_train_test(
        self, max_samples: int = 4000, force_recompute: bool = False
    ) -> Optional[dict]:
        """
        完整管线：加载 → 特征提取 → 返回训练/测试集。

        参数:
            force_recompute: 强制重新特征提取（忽略缓存）
        返回:
            {
                "X_train": (N, 158) ndarray,
                "y_train": (N,) ndarray,
                "X_test": (M, 158) ndarray,
                "y_test": (M,) ndarray,
                "feature_columns": list[str],
            }
            若数据不可用则返回 None。
        """
        if not force_recompute:
            cached_train = self.load_processed("train")
            cached_test = self.load_processed("test")
            if cached_train is not None and cached_test is not None:
                X_train, y_train = cached_train
                X_test, y_test = cached_test
                return {
                    "X_train": X_train, "y_train": y_train,
                    "X_test": X_test, "y_test": y_test,
                    "feature_columns": self._feature_columns,
                }

        df_train = self.load("train", max_samples=max_samples)
        df_test = self.load("test", max_samples=max(400, max_samples // 10))

        if df_train is None or df_test is None:
            logger.error("TeleAntiFraud 原始数据不可用")
            return None

        X_train, y_train = self.compute_features(df_train)
        X_test, y_test = self.compute_features(df_test)

        self.save_processed(X_train, y_train, "train")
        self.save_processed(X_test, y_test, "test")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_test": X_test, "y_test": y_test,
            "feature_columns": self._feature_columns,
        }


# ── 全局单例 ──────────────────────────────────────────────────
teleantifraud_loader = TeleAntiFraudLoader()
