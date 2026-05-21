"""
Domain-tuned speculative decoding (§3.4).

Implements:
  * Theoretical speedup formula (1 - α^(γ+1)) / (1 - α) [Leviathan et al. 2023]
  * Acceptance-rate measurement on a draft + target model
  * Speedup computation
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def theoretical_speedup(alpha: float, gamma: int) -> float:
    """Eq. from Leviathan et al. [9]:
        speedup(α, γ) = (1 - α^(γ+1)) / (1 - α).
    """
    if alpha <= 0:
        return 1.0
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1 - alpha ** (gamma + 1)) / (1 - alpha)


@dataclass
class SpeculativeStats:
    alpha: float        # Token acceptance rate
    gamma: int          # Draft tokens per round
    theoretical: float  # Theoretical speedup
    measured: float | None = None
    kv_cache_mb: float | None = None


def measure_acceptance_rate(
    draft_logits_seq: list,
    target_logits_seq: list,
    tolerance: float = 1.0,
) -> float:
    """
    Measure token acceptance rate when running greedy speculative decoding.
    The draft proposes tokens; target accepts if its top-1 matches.
    """
    matches, total = 0, 0
    for d, t in zip(draft_logits_seq, target_logits_seq):
        d_top = int(d.argmax(dim=-1))
        t_top = int(t.argmax(dim=-1))
        total += 1
        if d_top == t_top:
            matches += 1
    return matches / max(1, total)


def kv_cache_size(gamma: int, hidden_dim: int = 4096, n_layers: int = 24,
                  bytes_per_elem: int = 2) -> float:
    """Approximate KV cache footprint in MB for a draft horizon of γ tokens."""
    bytes_total = gamma * 2 * hidden_dim * n_layers * bytes_per_elem
    return bytes_total / (1024 * 1024)


def compute_all_gammas(alpha: float, gammas: list[int] = [3, 5, 7, 10]) -> list[SpeculativeStats]:
    """Compute theoretical + KV stats for a range of γ values."""
    return [
        SpeculativeStats(alpha=alpha, gamma=g,
                         theoretical=theoretical_speedup(alpha, g),
                         kv_cache_mb=kv_cache_size(g))
        for g in gammas
    ]
