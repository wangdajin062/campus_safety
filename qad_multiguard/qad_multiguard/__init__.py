"""
qad_multiguard — Reference implementation of the QAD-MultiGuard paper.

Modules:
    data            TeleAntiFraud-28k loader + AdvFraud-3k builder
    quantization    NVFP4 / Q4_K_M emulators + PTQ baselines
    distillation    Pure-KL QAD trainer with same-source teacher
    ov_freeze       Output-Variance Freeze regularizer
    privacy         128-d non-invertible F_v acoustic embedding + GLO attacks
    fusion          L-BFGS multimodal fusion (text/audio/url/meta)
    speculative     Domain-tuned speculative decoding
    metrics         Recovery rate, F1, KL div, GLO WER, etc.
    deployment      End-to-end pipeline (edge tier 1 + cloud tier 2 + fusion tier 3)
"""

__version__ = "1.0.0"
__author__ = "QAD-MultiGuard contributors"

from .config import Config, load_config

__all__ = ["Config", "load_config", "__version__"]
