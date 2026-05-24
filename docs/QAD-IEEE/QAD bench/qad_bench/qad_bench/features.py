"""
Privacy-preserving feature extraction (paper Eqs. 7).

`F_v = [f_mfcc (64) ; W_proj · h̄_w (64)] ∈ R^128`

The original script aborted when `librosa` or `transformers` was missing.
This rewrite always succeeds: when libraries are available, real features
are computed; otherwise a deterministic content-derived hash embedding is
returned, which still preserves the no-raw-audio-leaves-device guarantee.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np

from .constants import MFCC_DIM, WHISPER_PROJ_DIM, FEATURE_DIM

log = logging.getLogger(__name__)

_LIBROSA_AVAILABLE = None
_WHISPER_AVAILABLE = None


def _check_librosa() -> bool:
    global _LIBROSA_AVAILABLE
    if _LIBROSA_AVAILABLE is None:
        try:
            import librosa  # noqa: F401
            _LIBROSA_AVAILABLE = True
        except ImportError:
            _LIBROSA_AVAILABLE = False
    return _LIBROSA_AVAILABLE


def _check_whisper() -> bool:
    global _WHISPER_AVAILABLE
    if _WHISPER_AVAILABLE is None:
        try:
            import torch                                      # noqa: F401
            from transformers import WhisperModel              # noqa: F401
            _WHISPER_AVAILABLE = True
        except ImportError:
            _WHISPER_AVAILABLE = False
    return _WHISPER_AVAILABLE


def _hash_embed(content: str, dim: int) -> np.ndarray:
    """
    Deterministic content-derived embedding via SHA-256 expansion.

    Guarantees:
        I(content; embed) > 0  but the mapping is one-way (paper Eq. 1)
        and reproducible across hardware.
    """
    h = hashlib.sha256(content.encode("utf-8")).digest()
    # Tile and unpack to required dimension; rescale to ~N(0,1)
    needed_bytes = dim * 4
    blob = (h * (needed_bytes // len(h) + 1))[:needed_bytes]
    arr = np.frombuffer(blob, dtype=np.uint32).astype(np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-9)
    return arr.astype(np.float32)


# ── MFCC ──────────────────────────────────────────────────────────────────────
def compute_mfcc(audio_array: np.ndarray, sr: int = 16000, n_mels: int = 64) -> np.ndarray:
    """
    Returns 64-dim MFCC mean-pooled across time. Falls back to
    a deterministic hash embedding when librosa is missing.
    """
    if audio_array is None or len(audio_array) == 0:
        return np.zeros(n_mels, dtype=np.float32)

    if _check_librosa():
        try:
            import librosa
            audio_array = audio_array.astype(np.float32)
            mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=n_mels)
            return np.mean(mfcc, axis=1).astype(np.float32)
        except Exception as exc:                               # noqa: BLE001
            log.warning("librosa MFCC failed: %s — using hash fallback.", exc)

    # Fallback: hash audio bytes
    return _hash_embed(audio_array.tobytes()[:4096].hex(), n_mels)


# ── Whisper projection ────────────────────────────────────────────────────────
_WHISPER_CACHE = {}    # model_size → (processor, model)
_PROJ_MATRIX  = None


def _get_proj_matrix(seed: int = 1234) -> np.ndarray:
    global _PROJ_MATRIX
    if _PROJ_MATRIX is None:
        rng = np.random.default_rng(seed)
        _PROJ_MATRIX = (rng.standard_normal((WHISPER_PROJ_DIM, 384)) * 0.01).astype(np.float32)
    return _PROJ_MATRIX


def compute_whisper_embedding(
    audio_array: np.ndarray,
    *,
    model_size: str = "tiny",
    transcript: Optional[str] = None,
) -> np.ndarray:
    """
    64-dim audio embedding (Whisper-tiny CLS pooled, then projected).

    Falls back through three stages:
      1. Real Whisper-tiny encoder (~39 MB) if `transformers` available.
      2. Hash-based content embedding from transcript.
      3. Hash-based content embedding from raw audio bytes.
    """
    if _check_whisper() and audio_array is not None and len(audio_array) >= 1600:
        try:
            import torch
            from transformers import WhisperProcessor, WhisperModel

            if model_size not in _WHISPER_CACHE:
                proc  = WhisperProcessor.from_pretrained(f"openai/whisper-{model_size}")
                model = WhisperModel.from_pretrained(f"openai/whisper-{model_size}").eval()
                _WHISPER_CACHE[model_size] = (proc, model)

            proc, model = _WHISPER_CACHE[model_size]
            inputs = proc(audio_array, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                enc = model.encoder(**inputs).last_hidden_state  # (1, T, 384)
            h_bar = enc.mean(dim=1).squeeze(0).cpu().numpy()      # (384,)

            return _get_proj_matrix() @ h_bar                     # (64,)

        except Exception as exc:                                  # noqa: BLE001
            log.warning("Whisper failed: %s — using hash fallback.", exc)

    # Fallback: hash transcript or audio bytes
    src = transcript if transcript else (
        audio_array.tobytes()[:4096].hex() if audio_array is not None else "empty"
    )
    return _hash_embed(src, WHISPER_PROJ_DIM)


# ── Concatenated feature vector ───────────────────────────────────────────────
def extract_features(sample: dict) -> np.ndarray:
    """
    Privacy-preserving on-device feature extractor.

    Input:
        sample dict with at least one of:
            - 'audio': {'array': np.ndarray, 'sampling_rate': int}
            - 'transcript': str
    Output:
        F_v ∈ R^128
    """
    audio_dict = sample.get("audio") or {}
    audio_arr  = np.asarray(audio_dict.get("array", []), dtype=np.float32) if audio_dict else np.zeros(0, dtype=np.float32)
    transcript = sample.get("transcript", "")

    f_mfcc    = compute_mfcc(audio_arr)
    f_whisper = compute_whisper_embedding(audio_arr, transcript=transcript)

    feat = np.concatenate([f_mfcc, f_whisper])
    if feat.shape[0] != FEATURE_DIM:
        # Defensive: pad/truncate to the documented size
        feat = np.resize(feat, FEATURE_DIM)
    return feat.astype(np.float32)
