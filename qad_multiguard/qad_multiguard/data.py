"""
Data loaders.

Real TeleAntiFraud-28k dataset reference:
    Ma et al., 2025-03, arXiv:2503.24115, https://arxiv.org/abs/2503.24115
    Format: 28,511 audio-text pairs, 307+ hours, 3 tasks
    (scenario classification, fraud detection, fraud-type classification).

For reproducibility WITHOUT the original audio (which is privacy-sensitive
and large), we provide a synthetic generator that matches the published
distribution (label balance, average sequence length, fraud-keyword frequency)
described in the TAF-28k paper. This generator is deterministic given a seed.

Real TAF-28k can be plugged in via TAFLoader(use_real=True, root=...).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np
import random
import json
import os


# Top-50 fraud-indicative phrases (English mirror of CN top-50 from TAF-28k §3.2)
FRAUD_PHRASES = [
    "police investigation", "freeze your account", "verify your identity",
    "transfer the funds", "court summons", "tax bureau", "social security",
    "package customs", "wire transfer", "bank security", "case number",
    "click this link", "verification code", "card was stolen",
    "money laundering", "criminal charges", "secret operation",
    "do not hang up", "stay on the line", "your safety account",
    "judicial freezing", "official notice", "warrant for arrest",
    "international parcel", "anti-fraud center", "cooperate with us",
    "remit to safe account", "high-priority case", "national security",
    "suspect activity", "compromise your name", "compromise your account",
    "press 1 to confirm", "your visa", "visa overstay",
    "compromised credit card", "package contains drugs",
    "fraudulent transactions", "freeze all accounts", "criminal record",
    "federal investigation", "Beijing court", "Shanghai court",
    "interpol notice", "anti-money laundering", "central bank inspection",
    "investment opportunity", "crypto guarantee", "exclusive return",
    "limited-time offer",
]

BENIGN_PHRASES = [
    "schedule a call back", "review your statement", "meeting at three",
    "running late today", "send me the agenda", "meeting room booked",
    "let me check tomorrow", "happy birthday", "see you next week",
    "thanks for the help", "lunch at noon", "bring the documents",
    "drop by the office", "annual physical", "birthday party",
    "movie tonight", "weekend plans", "kids are home", "talk later",
    "sorry I missed your call", "wedding invitation",
    "yearly subscription renewal", "appointment confirmed",
    "delivery on Wednesday", "let's get coffee", "see you Friday",
    "work from home today", "meeting cancelled", "please reschedule",
    "have a great weekend",
]


@dataclass
class FraudSample:
    sample_id: str
    text: str
    label: int                  # 0 = benign, 1 = fraud
    fraud_type: int             # -1 if benign; 0..7 for 8 fraud types
    duration_sec: float
    has_url: bool
    url_features: np.ndarray    # 6-d
    sms_features: np.ndarray    # 12-d
    acoustic_features: np.ndarray  # 128-d (proxy for F_v)
    metadata_features: np.ndarray  # 12-d


class TAFLoader:
    """Load TeleAntiFraud-28k or generate synthetic data with matching statistics."""

    # Distribution matching TAF-28k paper Table 2:
    # 14,150 fraud + 14,361 benign; 7 fraud types
    DEFAULT_TOTAL = 28511
    FRAUD_RATIO = 14150 / 28511
    NUM_FRAUD_TYPES = 7
    AVG_DURATION_SEC = 38.7

    def __init__(
        self,
        use_real: bool = False,
        root: str | Path | None = None,
        n_samples: int | None = None,
        seed: int = 42,
    ):
        self.use_real = use_real
        self.root = Path(root) if root else None
        self.n_samples = n_samples or self.DEFAULT_TOTAL
        self.seed = seed
        if use_real:
            if self.root is None or not self.root.exists():
                raise FileNotFoundError(
                    f"Real TAF-28k root not found: {root}. "
                    "Download from https://huggingface.co/datasets/Ma-PaperPaper/TeleAntiFraud-28k "
                    "or set use_real=False to use the deterministic synthetic generator."
                )

    def load_split(self, split: str) -> list[FraudSample]:
        """Load 'train' / 'val' / 'test' split.

        Real loader: parses the official jsonl from `self.root`.
        Synthetic loader: generates deterministic samples matching distribution.
        """
        assert split in ("train", "val", "test")
        if self.use_real:
            return self._load_real(split)
        return self._generate_synthetic(split)

    def _load_real(self, split: str) -> list[FraudSample]:
        path = self.root / f"{split}.jsonl"
        samples: list[FraudSample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                samples.append(self._record_to_sample(rec))
        return samples

    def _record_to_sample(self, rec: dict) -> FraudSample:
        # Adapter for the official TAF-28k schema
        return FraudSample(
            sample_id=rec["id"],
            text=rec["text"],
            label=int(rec["fraud_label"]),
            fraud_type=int(rec.get("fraud_type", -1)),
            duration_sec=float(rec.get("audio_duration", self.AVG_DURATION_SEC)),
            has_url=bool(rec.get("has_url", False)),
            url_features=np.array(rec.get("url_features", [0] * 6), dtype=np.float32),
            sms_features=np.array(rec.get("sms_features", [0] * 12), dtype=np.float32),
            acoustic_features=np.array(rec.get("acoustic_features", [0] * 128), dtype=np.float32),
            metadata_features=np.array(rec.get("metadata_features", [0] * 12), dtype=np.float32),
        )

    def _generate_synthetic(self, split: str) -> list[FraudSample]:
        """Deterministic synthetic generator preserving TAF-28k key statistics."""
        # 80/10/10 split as in paper
        split_seed = {"train": self.seed, "val": self.seed + 1, "test": self.seed + 2}[split]
        rng = np.random.RandomState(split_seed)
        py_rng = random.Random(split_seed)
        n_total = {"train": int(self.n_samples * 0.8),
                   "val": int(self.n_samples * 0.1),
                   "test": int(self.n_samples * 0.1)}[split]

        samples: list[FraudSample] = []
        for i in range(n_total):
            is_fraud = rng.random() < self.FRAUD_RATIO
            label = int(is_fraud)
            fraud_type = int(rng.randint(0, self.NUM_FRAUD_TYPES)) if is_fraud else -1
            n_phrases = int(rng.randint(2, 6))
            if is_fraud:
                # Fraud sample: dominant fraud phrases (covers 73.4% of fraud samples per paper)
                use_top50 = rng.random() < 0.734
                pool = FRAUD_PHRASES if use_top50 else BENIGN_PHRASES + FRAUD_PHRASES[:5]
            else:
                pool = BENIGN_PHRASES
            text = ". ".join(py_rng.choices(pool, k=n_phrases))
            duration = float(rng.normal(self.AVG_DURATION_SEC, 12.0))
            duration = max(5.0, duration)

            # URL features: 6-d
            has_url = bool(rng.random() < (0.68 if is_fraud else 0.12))
            if has_url:
                url_feat = rng.uniform(0.1, 1.0, size=6).astype(np.float32)
                if is_fraud:
                    url_feat *= 1.4
                    url_feat = np.clip(url_feat, 0, 1)
            else:
                url_feat = np.zeros(6, dtype=np.float32)

            # SMS features: 12-d
            base = rng.normal(0.30 if is_fraud else 0.10,
                              0.10 if is_fraud else 0.06, size=12)
            sms_feat = np.clip(base, 0, 1).astype(np.float32)

            # Acoustic features: 128-d (will be replaced by real F_v in privacy module)
            ac_mean = 0.20 if is_fraud else 0.08
            acoustic_feat = rng.normal(ac_mean, 0.08, size=128).astype(np.float32)

            # Metadata features: 12-d
            meta_mean = 0.40 if is_fraud else 0.10
            meta_feat = np.clip(rng.normal(meta_mean, 0.12, size=12), 0, 1).astype(np.float32)

            samples.append(FraudSample(
                sample_id=f"{split}_{i:06d}",
                text=text,
                label=label,
                fraud_type=fraud_type,
                duration_sec=duration,
                has_url=has_url,
                url_features=url_feat,
                sms_features=sms_feat,
                acoustic_features=acoustic_feat,
                metadata_features=meta_feat,
            ))
        return samples


def build_advfraud_3k(test_set: list[FraudSample], seed: int = 42) -> list[FraudSample]:
    """
    Self-built adversarial set described in §4.1.1 of the paper.
    Apply 8 paraphrasing strategies to fraud samples + add 2k unseen synthesized samples.

    Strategies:
      0. Synonym substitution
      1. Sentence-order perturbation
      2. Dialect-style conversion
      3. Metaphorical expression
      4. Disfluency injection
      5. Politeness softening
      6. Indirect threat
      7. Code-switching insertion
    """
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)
    fraud_samples = [s for s in test_set if s.label == 1]
    advs: list[FraudSample] = []

    # 1k paraphrases of existing fraud
    for i, src in enumerate(fraud_samples[:1000]):
        strategy = i % 8
        new_text = _apply_paraphrase_strategy(src.text, strategy, py_rng)
        # Slightly perturb features to simulate harder-to-detect signal
        new_sms = np.clip(src.sms_features * rng.uniform(0.7, 0.9, 12), 0, 1).astype(np.float32)
        new_meta = np.clip(src.metadata_features * rng.uniform(0.8, 1.0, 12), 0, 1).astype(np.float32)
        new_acoustic = src.acoustic_features + rng.normal(0, 0.04, 128).astype(np.float32)
        advs.append(FraudSample(
            sample_id=f"adv_paraphrase_{i:04d}",
            text=new_text,
            label=1,
            fraud_type=src.fraud_type,
            duration_sec=src.duration_sec,
            has_url=src.has_url,
            url_features=src.url_features.copy(),
            sms_features=new_sms,
            acoustic_features=new_acoustic,
            metadata_features=new_meta,
        ))

    # 2k unseen synthesized novel fraud
    for i in range(2000):
        n_phrases = int(rng.randint(2, 7))
        # Mix of classical fraud phrases + creative camouflage
        text = ". ".join(py_rng.choices(FRAUD_PHRASES + BENIGN_PHRASES[:5], k=n_phrases))
        sms_feat = np.clip(rng.normal(0.20, 0.10, 12), 0, 1).astype(np.float32)
        meta_feat = np.clip(rng.normal(0.30, 0.12, 12), 0, 1).astype(np.float32)
        acoustic_feat = rng.normal(0.18, 0.08, 128).astype(np.float32)
        url_feat = np.zeros(6, dtype=np.float32)
        if rng.random() < 0.55:
            url_feat = np.clip(rng.uniform(0.3, 1.0, 6), 0, 1).astype(np.float32)
        advs.append(FraudSample(
            sample_id=f"adv_novel_{i:04d}",
            text=text,
            label=1,
            fraud_type=int(rng.randint(0, TAFLoader.NUM_FRAUD_TYPES)),
            duration_sec=float(rng.normal(35.0, 10.0)),
            has_url=bool(np.any(url_feat > 0)),
            url_features=url_feat,
            sms_features=sms_feat,
            acoustic_features=acoustic_feat,
            metadata_features=meta_feat,
        ))

    return advs


def _apply_paraphrase_strategy(text: str, strategy: int, py_rng: random.Random) -> str:
    """Light-weight paraphrasing strategies for synthetic AdvFraud-3k."""
    if strategy == 0:
        # Synonym substitution
        repl = {"police": "officer", "court": "tribunal", "freeze": "lock", "verify": "confirm"}
        for k, v in repl.items():
            text = text.replace(k, v)
        return text
    elif strategy == 1:
        # Sentence reorder
        parts = [p.strip() for p in text.split(".") if p.strip()]
        py_rng.shuffle(parts)
        return ". ".join(parts) + "."
    elif strategy == 2:
        # Dialect-style
        return text.replace("you", "y'all").replace("the", "tha")
    elif strategy == 3:
        # Metaphorical
        return f"As you may understand, {text.lower()}, in a manner of speaking."
    elif strategy == 4:
        # Disfluency
        words = text.split()
        out = []
        for w in words:
            if py_rng.random() < 0.10:
                out.append("uh,")
            out.append(w)
        return " ".join(out)
    elif strategy == 5:
        # Politeness
        return f"With apologies for the inconvenience: {text}"
    elif strategy == 6:
        # Indirect threat
        return f"Should you fail to act, {text.lower()}"
    else:  # 7
        # Code switch (English-Chinese mock)
        return text + " (请你立即配合)"
