"""
QAD distillation trainer.

Implements pure KL divergence distillation (the paper's loss) plus the four
ablation variants from Table IV:
  - pure_kl       (eq. 1, the paper's main loss)
  - mse           (MSE on logits)
  - cross_entropy (= QAT, included as ablation)
  - three_term    (early-version mixed loss; weak baseline)
  - kl_task_reg   (KL + task regularizer)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainerStats:
    step: int = 0
    loss: float = 0.0
    kl_div_vs_teacher: float = 0.0
    ov_active: bool = False
    grad_norm: float = 0.0


class DistillationTrainer:
    """
    Implements pure KL distillation with optional OV-Freeze.

    The trainer is loss-agnostic — choose `loss_fn` from the registry below.
    For ablation studies, simply switch `loss_fn`.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        loss: str = "pure_kl",
        learning_rate: float = 1e-5,
        warmup_steps: int = 100,
        total_steps: int = 2000,
        softmax_temp: float = 1.0,
        ov_freeze_layers: dict | None = None,
        ov_freeze_step_ratio: float = 0.30,
        device: str = "cpu",
    ):
        self.student = student.to(device)
        self.teacher = teacher.to(device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.loss_name = loss
        self.loss_fn = LOSS_REGISTRY[loss]
        self.softmax_temp = softmax_temp
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.peak_lr = learning_rate
        self.device = device
        self.optimizer = torch.optim.AdamW(
            (p for p in self.student.parameters() if p.requires_grad),
            lr=learning_rate, weight_decay=0.0,
        )
        self.ov_freeze_layers = ov_freeze_layers or {}
        self.ov_freeze_step_ratio = ov_freeze_step_ratio
        self.history: list[TrainerStats] = []

    def _get_lr(self, step: int) -> float:
        """Linear warmup → cosine decay schedule (NVIDIA QAD §3.4)."""
        if step < self.warmup_steps:
            return self.peak_lr * (step / max(1, self.warmup_steps))
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * self.peak_lr * (1 + math.cos(math.pi * progress))

    def _set_lr(self, lr: float) -> None:
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _maybe_activate_ov(self, step: int) -> bool:
        if not self.ov_freeze_layers:
            return False
        from .ov_freeze import schedule_ov_freeze
        return schedule_ov_freeze(self.ov_freeze_layers, step, self.total_steps,
                                  self.ov_freeze_step_ratio)

    def step(self, batch: torch.Tensor, labels: torch.Tensor | None = None) -> TrainerStats:
        """One training step.

        Args:
            batch: tensor of shape [B, ...], ready for student/teacher forward()
            labels: optional ground-truth labels (only used by losses with task term)
        """
        step_id = len(self.history)
        # Schedule
        self._set_lr(self._get_lr(step_id))
        ov_active = self._maybe_activate_ov(step_id)

        # Forward
        student_logits = self.student(batch)
        with torch.no_grad():
            teacher_logits = self.teacher(batch)

        # Loss
        loss, kl_div = self.loss_fn(
            student_logits, teacher_logits, labels=labels, temp=self.softmax_temp
        )

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        # Clip
        grad_norm = float(torch.nn.utils.clip_grad_norm_(
            self.student.parameters(), max_norm=1.0
        ).item())
        self.optimizer.step()

        stats = TrainerStats(
            step=step_id,
            loss=float(loss.item()),
            kl_div_vs_teacher=float(kl_div),
            ov_active=ov_active,
            grad_norm=grad_norm,
        )
        self.history.append(stats)
        return stats


# ─── Losses ─────────────────────────────────────────────────────────────
def _measure_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                temp: float = 1.0) -> torch.Tensor:
    """KL(p_T || p_S)."""
    p_T = F.softmax(teacher_logits / temp, dim=-1)
    log_p_S = F.log_softmax(student_logits / temp, dim=-1)
    return F.kl_div(log_p_S, p_T, reduction="batchmean") * (temp ** 2)


def loss_pure_kl(s_logits, t_logits, labels=None, temp=1.0):
    """Eq. (1) of the paper: pure KL divergence."""
    kl = _measure_kl(s_logits, t_logits, temp)
    return kl, float(kl.item())


def loss_mse(s_logits, t_logits, labels=None, temp=1.0):
    """MSE on logits (Table IV ablation row 2)."""
    mse = F.mse_loss(s_logits, t_logits)
    kl = _measure_kl(s_logits, t_logits, temp)
    return mse, float(kl.item())


def loss_cross_entropy(s_logits, t_logits, labels=None, temp=1.0):
    """Cross-entropy (= QAT). Falls back to teacher argmax if labels missing."""
    if labels is None:
        labels = t_logits.argmax(dim=-1)
    ce = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), labels.reshape(-1))
    kl = _measure_kl(s_logits, t_logits, temp)
    return ce, float(kl.item())


def loss_three_term(s_logits, t_logits, labels=None, temp=1.0):
    """Early-version three-term hybrid (task + KD + quantization regularizer)."""
    if labels is None:
        labels = t_logits.argmax(dim=-1)
    task = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), labels.reshape(-1))
    kd = _measure_kl(s_logits, t_logits, temp)
    # Quantization regularizer (rough proxy: variance of logits)
    quant_reg = s_logits.var(dim=-1).mean()
    total = 0.5 * task + 0.4 * kd + 0.1 * quant_reg
    return total, float(kd.item())


def loss_kl_task_reg(s_logits, t_logits, labels=None, temp=1.0):
    """KL + task regularizer (Table IV row 5)."""
    kl = _measure_kl(s_logits, t_logits, temp)
    if labels is None:
        labels = t_logits.argmax(dim=-1)
    task = F.cross_entropy(s_logits.reshape(-1, s_logits.shape[-1]), labels.reshape(-1))
    total = 0.9 * kl + 0.1 * task
    return total, float(kl.item())


LOSS_REGISTRY: dict[str, Callable] = {
    "pure_kl":        loss_pure_kl,
    "mse":            loss_mse,
    "cross_entropy":  loss_cross_entropy,
    "three_term":     loss_three_term,
    "kl_task_reg":    loss_kl_task_reg,
}
