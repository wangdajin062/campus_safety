"""
_app_data.py — Safe import helper for campus_safety_v3 modules.

Provides fallback constants and a helper to add the campus_safety_v3 backend
path to ``sys.path``.  All imports from the companion app are wrapped in
try/except so that every symbol has a usable default even when the app tree
is not deployed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── campus_safety_v3 backend root ──────────────────────────────────────
_CAMPUS_PATH = Path("d:/campus_safety_v3_complete/backend")

# ── Safe defaults (used when campus_safety_v3 is unavailable) ──────────
STUDENT_ARCH    = None
ALPHA_TUNED     = 0.86
GAMMA           = 5

W_TEXT          = 0.40
W_AUDIO         = 0.30
W_URL           = 0.20
W_META          = 0.10
FUSION_SCALE    = 5.0
FUSION_BIAS     = 0.0

# Tracks whether ensure_import() successfully loaded ALL modules
_import_successful = False

MFCC_DIM        = 64
EMBEDDING_DIM   = 128
SAMPLE_RATE     = 16_000
N_MELS          = 64

APP_QAD_CONFIG  = None


def ensure_import() -> bool:
    """Add ``_CAMPUS_PATH`` to ``sys.path`` and attempt to import
    campus_safety_v3 constants, falling back to the module-level defaults
    on failure.

    Returns ``True`` when the app modules were loaded successfully,
    ``False`` otherwise.
    """
    import warnings

    if not _CAMPUS_PATH.exists():
        warnings.warn(f"campus_safety_v3 backend not found at {_CAMPUS_PATH}")
    elif str(_CAMPUS_PATH) not in sys.path:
        sys.path.insert(0, str(_CAMPUS_PATH))

    # Track whether ALL four import groups succeed
    _all_ok = True

    # pylint: disable=import-outside-toplevel
    try:
        from ml.speculative_decoder import STUDENT_ARCH as _STUDENT_ARCH
        from ml.speculative_decoder import ALPHA_TUNED as _ALPHA_TUNED
        from ml.speculative_decoder import GAMMA as _GAMMA
    except ImportError:
        warnings.warn("ml.speculative_decoder import failed; using defaults")
        _all_ok = False
    else:
        globals().update(
            STUDENT_ARCH=_STUDENT_ARCH,
            ALPHA_TUNED=_ALPHA_TUNED,
            GAMMA=_GAMMA,
        )

    try:
        from ml.qad_pipeline import QADConfig as _AppQADConfig
    except ImportError:
        warnings.warn("ml.qad_pipeline import failed; using defaults")
        _all_ok = False
    else:
        globals().update(APP_QAD_CONFIG=_AppQADConfig)

    try:
        from ml.multimodal_detector import W_TEXT as _W_TEXT
        from ml.multimodal_detector import W_AUDIO as _W_AUDIO
        from ml.multimodal_detector import W_URL as _W_URL
        from ml.multimodal_detector import W_META as _W_META
        from ml.multimodal_detector import FUSION_BIAS as _FUSION_BIAS
        from ml.multimodal_detector import FUSION_SCALE as _FUSION_SCALE
    except ImportError:
        warnings.warn("ml.multimodal_detector import failed; using defaults")
        _all_ok = False
    else:
        globals().update(
            W_TEXT=_W_TEXT,
            W_AUDIO=_W_AUDIO,
            W_URL=_W_URL,
            W_META=_W_META,
            FUSION_BIAS=_FUSION_BIAS,
            FUSION_SCALE=_FUSION_SCALE,
        )

    try:
        from ml.acoustic_embedding import MFCC_DIM as _MFCC_DIM
        from ml.acoustic_embedding import EMBEDDING_DIM as _EMBEDDING_DIM
        from ml.acoustic_embedding import SAMPLE_RATE as _SAMPLE_RATE
        from ml.acoustic_embedding import N_MELS as _N_MELS
    except ImportError:
        warnings.warn("ml.acoustic_embedding import failed; using defaults")
        _all_ok = False
    else:
        globals().update(
            MFCC_DIM=_MFCC_DIM,
            EMBEDDING_DIM=_EMBEDDING_DIM,
            SAMPLE_RATE=_SAMPLE_RATE,
            N_MELS=_N_MELS,
        )

    global _import_successful
    _import_successful = _all_ok
    return _import_successful
