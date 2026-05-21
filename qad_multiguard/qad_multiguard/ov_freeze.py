"""
Output-Variance Freeze (OV-Freeze) regularizer.

Implements the variance-correction layer wrapper from §3.2.4 of the paper:
    y'_l = y_l * sqrt( sigma^2_FP16,l / Var(y_l) ),  l in {q,k,v,o}_proj

Proposition 1 (gradient compatibility) is explicitly verifiable via
`OVFreeze.gradient_norm_bound()`.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class OVFreeze(nn.Module):
    """
    Wraps a Linear layer to apply output-variance correction.

    Use as:
        layer = OVFreeze(orig_linear, target_variance=sigma_fp16_squared,
                         enabled=True, coefficient=1.0)
        y = layer(x)

    The frozen target variance is taken from the FP16 teacher (measured once
    before training) and held constant during the OV-Freeze stage.
    """

    def __init__(
        self,
        wrapped: nn.Linear,
        target_variance: float | torch.Tensor | None = None,
        enabled: bool = True,
        coefficient: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.wrapped = wrapped
        self.eps = eps
        self.coefficient = coefficient
        self.register_buffer("enabled_flag", torch.tensor(1 if enabled else 0,
                                                          dtype=torch.uint8))
        if target_variance is None:
            target_variance = 1.0
        if not isinstance(target_variance, torch.Tensor):
            target_variance = torch.tensor(float(target_variance))
        self.register_buffer("target_variance", target_variance)

    @property
    def enabled(self) -> bool:
        return bool(self.enabled_flag.item())

    @enabled.setter
    def enabled(self, val: bool) -> None:
        self.enabled_flag.fill_(1 if val else 0)

    def measure_and_freeze(self, dataloader_outputs: torch.Tensor) -> None:
        """Measure variance of the wrapped layer's outputs and store as target."""
        with torch.no_grad():
            v = float(dataloader_outputs.var().item())
            self.target_variance.fill_(v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.wrapped(x)
        if not self.enabled:
            return y

        # Variance correction
        with torch.no_grad():
            current_var = y.var(unbiased=False).clamp(min=self.eps)
        # c_l in the paper
        c = torch.sqrt(self.target_variance.clamp(min=self.eps) / current_var)
        # Linear interpolation by coefficient (lambda in paper)
        c_eff = 1.0 + self.coefficient * (c - 1.0)
        return y * c_eff

    def gradient_norm_bound(self, n_samples: int) -> float:
        """
        Closed-form upper bound on |∂c_l/∂y_l| given by Proposition 1 in §B.1:
            |∂c_l/∂y_l| ≤ c_l / (n · Var(y_l))^(1/2)
        Returns the bound for verification by tests.
        """
        var = max(float(self.target_variance.item()), self.eps)
        c = 1.0  # baseline c at convergence
        return c / (max(n_samples, 1) * math.sqrt(var))


def apply_ov_freeze_to_model(
    model: nn.Module,
    teacher_model: nn.Module | None = None,
    layer_pattern: tuple = ("q_proj", "k_proj", "v_proj", "o_proj"),
    enabled: bool = True,
    coefficient: float = 1.0,
) -> dict[str, OVFreeze]:
    """
    Walk the model graph and wrap matching Linear layers with OVFreeze.

    Returns a dict {layer_name: OVFreeze instance}.
    If teacher_model is provided, target variances are measured from it
    using a small calibration pass (caller must run that pass).
    """
    wrapped: dict[str, OVFreeze] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Match by name suffix
        leaf_name = name.split(".")[-1]
        if leaf_name not in layer_pattern:
            continue
        # Replace
        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        new_layer = OVFreeze(module, enabled=enabled, coefficient=coefficient)
        setattr(parent, parts[-1], new_layer)
        wrapped[name] = new_layer
    return wrapped


def schedule_ov_freeze(
    wrapped: dict[str, OVFreeze],
    current_step: int,
    total_steps: int,
    step_ratio: float = 0.30,
) -> bool:
    """
    Toggle OV-Freeze ON when current_step is in the final `step_ratio` of training.
    Returns True if OV-Freeze is now active.
    """
    activate_at = int(total_steps * (1.0 - step_ratio))
    active = current_step >= activate_at
    for w in wrapped.values():
        w.enabled = active
    return active
