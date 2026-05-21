"""Configuration management for QAD-MultiGuard."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json
import os



@dataclass
class QuantConfig:
    """Quantization configuration."""
    format: str = "nvfp4"           # 'nvfp4' | 'q4_k_m' | 'ptq' | 'awq' | 'gptq' | 'spinquant' | 'quarot' | 'bitdistiller'
    block_size: int = 16            # NVFP4 block size
    use_e4m3_scale: bool = True     # FP8 E4M3 scale factor (NVFP4 only)
    bits: int = 4                   # Effective bit-width


@dataclass
class QADConfig:
    """QAD training configuration (matches NVIDIA NVFP4 QAD §3.4)."""
    teacher_model: str = "qwen2.5-0.5b-instruct"  # Same-source teacher
    student_quant: QuantConfig = field(default_factory=QuantConfig)
    loss: str = "pure_kl"           # 'pure_kl' (preferred) | 'mse' | 'cross_entropy' | 'three_term' | 'kl_task_reg'
    softmax_temp: float = 1.0       # Temperature: 1.0 in paper
    learning_rate: float = 1e-5     # Cosine decay
    warmup_steps: int = 100
    batch_size: int = 8
    seq_length: int = 4096
    total_steps: int = 2000
    total_tokens: int = int(0.5e9)  # 0.5B (~1.7% of original SFT data)
    # ── App-aligned hyper-parameters ──────────────────────────────────────
    alpha: float = 0.4
    beta: float = 0.5
    gamma_coeff: float = 0.1
    fp16_ppl: float = 8.43
    int4_ptq_ppl: float = 9.42
    int4_qad_ppl: float = 8.73
    int4_ov_ppl: float = 8.62
    int4_size_mb: int = 240
    fp16_size_mb: int = 960
    hidden_dim: int = 896
    n_layers: int = 24
    vocab_size: int = 151936


@dataclass
class OVFreezeConfig:
    """Output-Variance Freeze regularizer configuration."""
    enabled: bool = True
    layers: tuple = ("o_proj", "v_proj", "q_proj", "k_proj")
    step_ratio: float = 0.30        # Final 30% of training steps
    coefficient: float = 1.0        # λ in the loss; default 1.0


@dataclass
class PrivacyConfig:
    """Privacy-preserving acoustic embedding configuration."""
    n_mfcc: int = 64
    sample_rate: int = 16000
    hop_length_ms: int = 10
    n_fft_ms: int = 25
    n_mels: int = 64
    whisper_model: str = "tiny"     # 384-d output
    proj_dim: int = 64              # W_proj : R^(64x384)
    embed_dim: int = 128            # f_mfcc + W_proj h_w
    add_dp_noise: bool = False      # (ε=1.5, δ=10^-5)-DP if True
    dp_epsilon: float = 1.5
    dp_delta: float = 1e-5
    dp_sensitivity: float = 2.0


@dataclass
class FusionConfig:
    """Multimodal risk-fusion configuration."""
    weights: tuple = (0.40, 0.30, 0.20, 0.10)  # text, audio, url, meta
    bias: float = 0.0
    scale: float = 5.0
    high_risk_threshold: float = 0.70
    medium_risk_threshold: float = 0.35


@dataclass
class SpeculativeConfig:
    """Speculative decoding configuration."""
    draft_model: str = "qwen2-0.1b-tuned"      # Domain-tuned 124M
    gamma: int = 5                  # Draft tokens per round
    target_acceptance_rate: float = 0.86       # After domain tuning
    student_arch: dict = field(default_factory=lambda: {
        "backbone": "Qwen2.5-0.5B-Instruct",
        "params_fp16_M": 494,
        "size_int4_MB": 240,
        "hidden_dim": 896,
        "n_layers": 24,
        "attn_heads_Q": 14,
        "attn_heads_KV": 2,
        "ffn_dim": 4864,
        "vocab_size": 151936,
        "quant_scheme": "Q4_K_M",
    })


@dataclass
class Config:
    """Top-level configuration."""
    seed: int = 42
    qad: QADConfig = field(default_factory=QADConfig)
    ov_freeze: OVFreezeConfig = field(default_factory=OVFreezeConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    speculative: SpeculativeConfig = field(default_factory=SpeculativeConfig)
    output_dir: str = "./runs"
    data_dir: str = "./data"
    device: str = "cpu"             # 'cpu' | 'cuda' (auto-fallback to cpu)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load configuration. If path is None or missing, returns defaults."""
    if path is None or not os.path.exists(path):
        return Config()
    with open(path, "r") as f:
        data = json.load(f)
    cfg = Config()
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg
