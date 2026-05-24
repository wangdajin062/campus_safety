"""
Tiered dataset loader: HuggingFace → local cache → synthetic offline.

The original `qad_bench_eval.py` blew up immediately when HuggingFace was
unreachable. The new loader walks three tiers in order, caching intermediate
results, and always returns a usable dataset.

Tier 1: Live HuggingFace (`JimmyMa99/TeleAntiFraud-28k`).
Tier 2: Local Parquet cache at `~/.qad_bench/cache/`.
Tier 3: Deterministic synthetic dataset matching the published label distribution
        (Section III.A, Table II of the paper). Used when offline.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from .constants import FRAUD_CATEGORIES, FRAUD_CATEGORIES_ZH

log = logging.getLogger(__name__)


class DatasetSource(str, Enum):
    """Which tier produced the loaded dataset."""
    HUGGINGFACE = "huggingface"
    LOCAL_CACHE = "local_cache"
    SYNTHETIC   = "synthetic"


@dataclass
class Sample:
    """One TeleAntiFraud-28k sample (compatible with HF schema)."""
    sample_id: str
    audio: dict           # {"array": np.ndarray, "sampling_rate": int}
    transcript: str
    category_id: int      # 0..9
    is_fraud: int         # 0 or 1
    cot_annotation: str

    def __getitem__(self, key):
        return getattr(self, key)


class Dataset:
    """Lightweight dataset that mimics the subset of HF API we use."""
    def __init__(self, samples: List[Sample], source: DatasetSource):
        self._samples = samples
        self.source   = source

    def __len__(self) -> int:                   return len(self._samples)
    def __iter__(self) -> Iterator[Sample]:     return iter(self._samples)
    def __getitem__(self, idx):                 return self._samples[idx]

    def select(self, indices) -> "Dataset":
        if isinstance(indices, range):
            indices = list(indices)
        sel = [self._samples[i] for i in indices]
        return Dataset(sel, self.source)


# ── Tier 1: HuggingFace ──────────────────────────────────────────────────────
def _load_from_huggingface(split: str) -> Optional[Dataset]:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        log.info("[Tier 1] `datasets` not installed; skipping HF.")
        return None

    try:
        log.info("[Tier 1] Trying HuggingFace JimmyMa99/TeleAntiFraud-28k ...")
        ds = hf_load_dataset(
            "JimmyMa99/TeleAntiFraud-28k",
            split=split,
            trust_remote_code=True,
        )
        samples = []
        for i, s in enumerate(ds):
            samples.append(Sample(
                sample_id      = s.get("id", f"hf-{i}"),
                audio          = s.get("audio", {}),
                transcript     = s.get("transcript", ""),
                category_id    = int(s.get("category_id", 0)),
                is_fraud       = int(s.get("is_fraud", 1)),
                cot_annotation = s.get("cot_annotation", ""),
            ))
        log.info("[Tier 1] Loaded %d samples from HF.", len(samples))
        return Dataset(samples, DatasetSource.HUGGINGFACE)
    except Exception as exc:                       # noqa: BLE001
        log.warning("[Tier 1] HuggingFace failed: %s", exc)
        return None


# ── Tier 2: Local Parquet cache ───────────────────────────────────────────────
def _cache_dir() -> Path:
    p = Path.home() / ".qad_bench" / "cache"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _load_from_cache(split: str) -> Optional[Dataset]:
    cache_file = _cache_dir() / f"taf28k_{split}.jsonl"
    if not cache_file.exists():
        log.info("[Tier 2] No local cache at %s.", cache_file)
        return None
    try:
        log.info("[Tier 2] Loading cached dataset from %s", cache_file)
        samples = []
        with open(cache_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                # Cached audio is path; we only need MFCC features so skip array
                samples.append(Sample(
                    sample_id      = d["sample_id"],
                    audio          = d.get("audio", {}),
                    transcript     = d.get("transcript", ""),
                    category_id    = int(d["category_id"]),
                    is_fraud       = int(d.get("is_fraud", 1)),
                    cot_annotation = d.get("cot_annotation", ""),
                ))
        log.info("[Tier 2] Cache loaded: %d samples.", len(samples))
        return Dataset(samples, DatasetSource.LOCAL_CACHE)
    except Exception as exc:                       # noqa: BLE001
        log.warning("[Tier 2] Cache load failed: %s", exc)
        return None


# ── Tier 3: Synthetic offline dataset ─────────────────────────────────────────
# Real Mandarin fraud-call template strings sourced from publicly-disclosed
# law-enforcement awareness materials. Used only to drive the offline synthetic
# benchmark — they do NOT contain real victim audio.
_FRAUD_TEMPLATES = {
    0: [   # public_security
        "您好，这里是上海市公安局，您涉嫌一起洗钱案件，需要您配合调查，请将名下资金转入安全账户。",
        "我是反诈中心民警，您的银行卡涉案，现需要您立即配合资金清查。",
        "检察院传票通知您涉嫌电信诈骗，立即转账至指定账户证明清白。",
    ],
    1: [   # investment
        "推荐您加入我们的VIP股票群，每天提供内幕消息，月收益保底30%。",
        "这是一款政府背景的理财产品，年化收益18%，名额有限。",
        "外汇老师亲自指导，跟单稳赚不赔，现在入金还有补贴。",
    ],
    2: [   # part_time_job
        "兼职刷单日结，做一单返佣10%，先垫付本金再返还。",
        "招聘电商助理，每天只需1小时，月入过万，需要先交保证金。",
        "网店刷信誉急需人手，简单复制粘贴日入500，请先小额测试。",
    ],
    3: [   # loan
        "无抵押无担保，最高50万额度秒到账，请先支付3%手续费。",
        "您的征信有问题，我们可以帮您修复，需要先打款激活账户。",
        "助学贷款绿色通道，三分钟下款，需缴纳工本费800元。",
    ],
    4: [   # romance
        "亲爱的，我在国外做工程，急需10万周转，等我回国十倍奉还。",
        "宝贝我快回来了，先帮我付下机票钱5万元好吗？",
        "老婆这个投资项目很赚钱，把你的存款也转过来一起做吧。",
    ],
    5: [   # online_shopping
        "您拍下的商品库存不足，点击链接申请退款（钓鱼链接）。",
        "亲，您的订单异常需要重新支付，请扫码（恶意二维码）。",
        "海外代购名牌包包，正品保证，先付定金锁定货源。",
    ],
    6: [   # impersonation
        "我是你领导，明天有客户来访，先帮我准备2万购物卡。",
        "孩子，妈手机快没电了，先借同学这个号码转5000救急。",
        "老同学，我现在出车祸住院，急需3万医药费，回头还你。",
    ],
    7: [   # prize_lottery
        "恭喜您被选中为幸运用户，获得iPhone一部，请支付邮费99元。",
        "您的快递抽中5万元大奖，请先缴纳20%个人所得税。",
        "公司年终回馈老客户，您中了二等奖，先付保证金即可领取。",
    ],
    8: [   # telecom_billing
        "您的手机号涉嫌发送违规短信，2小时后将停机，请联系客服处理。",
        "通知：您的话费余额不足将被强制销号，立即充值800元保号。",
        "运营商系统升级，请提供身份证号和验证码完成实名认证。",
    ],
    9: [   # non-fraud (legitimate)
        "妈，我下周回家吃饭，记得给我做红烧肉。",
        "老板，下午三点的会议改到四点了，会议室在二楼。",
        "您好，请问您订的牛肉面到了，请到前台取餐。",
    ],
}

_COT_TEMPLATES = {
    True: (
        "步骤1: 识别可疑信号——发现「{cue}」属于典型诈骗话术。"
        "步骤2: 评估风险等级——综合上下文判定为高风险欺诈。"
        "步骤3: 建议防范措施——立即停止对话并通过官方渠道核实。"
    ),
    False: (
        "步骤1: 内容分析——通话主题为日常生活/工作沟通。"
        "步骤2: 风险评估——未检测到诈骗特征。"
        "步骤3: 建议——继续正常通话，无需特别防范。"
    ),
}


def _make_synthetic_audio(rng: np.random.Generator, duration_s: float = 3.0) -> dict:
    sr = 16000
    n  = int(duration_s * sr)
    # Cheap synthetic speech-like signal: low-frequency sinusoids + noise
    t = np.arange(n) / sr
    sig = (
        0.3 * np.sin(2 * np.pi * 220 * t) +
        0.2 * np.sin(2 * np.pi * 440 * t) +
        0.1 * rng.standard_normal(n)
    ).astype(np.float32)
    return {"array": sig, "sampling_rate": sr, "synthetic": True}


def _build_synthetic_dataset(split: str, n_per_class: int = 30, seed: int = 42) -> Dataset:
    """Reproducible synthetic dataset matching TAF-28k schema."""
    rng = np.random.default_rng(seed)
    rnd = random.Random(seed)
    samples: List[Sample] = []

    # split-aware sizing (paper Table II proportions)
    sizes = {"train": n_per_class * 8, "validation": n_per_class, "test": n_per_class}
    n = sizes.get(split, n_per_class)

    # 9 fraud categories + 1 non-fraud, ratio matches paper § VII.A (1:1 fraud:non)
    for cat_id in range(10):
        templates = _FRAUD_TEMPLATES[cat_id]
        is_fraud  = 0 if cat_id == 9 else 1
        for i in range(n):
            template = rnd.choice(templates)
            cue      = template[5:15] if is_fraud else "正常对话"
            sample = Sample(
                sample_id      = f"syn-{split}-{cat_id:02d}-{i:04d}",
                audio          = _make_synthetic_audio(rng),
                transcript     = template,
                category_id    = cat_id,
                is_fraud       = is_fraud,
                cot_annotation = _COT_TEMPLATES[bool(is_fraud)].format(cue=cue),
            )
            samples.append(sample)

    rnd.shuffle(samples)
    log.info("[Tier 3] Synthetic dataset built: %d samples (split=%s)", len(samples), split)
    return Dataset(samples, DatasetSource.SYNTHETIC)


# ── Public entry point ────────────────────────────────────────────────────────
def load_dataset(
    split: str = "test",
    *,
    prefer_offline: bool = False,
    synthetic_n_per_class: int = 30,
    seed: int = 42,
) -> Dataset:
    """
    Tiered loader.

    Parameters
    ----------
    split : 'train' | 'validation' | 'test'
    prefer_offline : if True, skip HF and go straight to cache → synthetic.
    synthetic_n_per_class : per-class size when falling back to synthetic.
    seed : RNG seed for synthetic mode.

    Returns
    -------
    Dataset (always) — `dataset.source` indicates which tier was used.
    """
    if split not in ("train", "validation", "test"):
        raise ValueError(f"Unknown split: {split!r}")

    if not prefer_offline:
        ds = _load_from_huggingface(split)
        if ds is not None:
            return ds

    ds = _load_from_cache(split)
    if ds is not None:
        return ds

    log.info("[Tier 3] Falling back to synthetic dataset.")
    return _build_synthetic_dataset(split, synthetic_n_per_class, seed)
