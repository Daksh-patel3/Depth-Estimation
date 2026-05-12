"""LoRA injection helpers for Depth Anything V2.

Wraps the HF ``Depth-Anything-V2-Large-hf`` model with PEFT LoRA adapters on
the DINOv2 attention query/value projections. The decoder/head and original
backbone weights stay frozen, so only the LoRA adapters get trained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from peft import LoraConfig, PeftModel, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# Names of attention projection submodules inside DINOv2 attention blocks.
# In HF transformers, these live as ``...attention.query`` / ``...attention.value``.
DEFAULT_TARGETS: Tuple[str, ...] = ("query", "value")


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable, frozen) parameter counts for ``model``."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable, frozen


def _print_param_summary(model: nn.Module, label: str = "model") -> None:
    """Print a short report of trainable vs frozen parameters."""
    trainable, frozen = count_parameters(model)
    total = trainable + frozen
    pct = 100.0 * trainable / max(total, 1)
    print(
        f"[{label}] trainable={trainable:,} ({pct:.4f}%) | "
        f"frozen={frozen:,} | total={total:,}"
    )


def _resolve_target_module_names(
    model: nn.Module,
    target_keywords: Sequence[str] = DEFAULT_TARGETS,
) -> List[str]:
    """Find linear submodules whose name ends with any of ``target_keywords``.

    PEFT requires either explicit module names or a regex; we discover them
    dynamically because the exact prefix changes between HF versions of the
    DINOv2 backbone (``backbone.encoder.layer.X.attention.attention.query`` in
    older releases vs ``...layer.X.attention.query`` in newer ones).
    """
    names: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in target_keywords:
            names.append(name)
    return names


def get_lora_model(
    base_model: nn.Module,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_keywords: Sequence[str] = DEFAULT_TARGETS,
    verbose: bool = True,
):
    """Wrap ``base_model`` with PEFT LoRA adapters on attention q/v projections.

    Parameters
    ----------
    base_model: nn.Module
        A pretrained Depth Anything V2 model (HF ``AutoModelForDepthEstimation``).
    rank, alpha, dropout: float / int
        Standard LoRA hyperparameters.
    target_keywords: tuple of str
        Submodule leaf names to inject adapters into.
    verbose: bool
        Print a parameter summary before / after wrapping.

    Returns
    -------
    PeftModel | nn.Module
        The PEFT-wrapped model. If PEFT is not installed, falls back to a
        manual freeze + LoRA reimplementation so this module stays runnable.
    """
    if verbose:
        _print_param_summary(base_model, label="base (pre-LoRA)")

    target_names = _resolve_target_module_names(base_model, target_keywords)
    if not target_names:
        raise RuntimeError(
            f"No linear modules matched target_keywords={target_keywords!r}. "
            f"Inspect the model's named_modules() to add the right names."
        )

    if PEFT_AVAILABLE:
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=target_names,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        # Freeze base weights first; PEFT will mark adapters trainable.
        for p in base_model.parameters():
            p.requires_grad = False
        peft_model = get_peft_model(base_model, config)
        if verbose:
            _print_param_summary(peft_model, label=f"peft (rank={rank})")
        return peft_model

    # Fallback path — only triggers if PEFT failed to import. Implements a
    # minimal, self-contained LoRA wrapper to keep the file runnable.
    print("[lora_wrapper] PEFT not available — using manual fallback.")
    for p in base_model.parameters():
        p.requires_grad = False
    for name in target_names:
        module = dict(base_model.named_modules())[name]
        if not isinstance(module, nn.Linear):
            continue
        wrapped = _ManualLoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        parent_name, leaf_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = base_model if parent_name == "" else dict(base_model.named_modules())[parent_name]
        setattr(parent, leaf_name, wrapped)
    if verbose:
        _print_param_summary(base_model, label=f"manual-lora (rank={rank})")
    return base_model


class _ManualLoRALinear(nn.Module):
    """Tiny LoRA adapter wrapping an nn.Linear (used only without PEFT)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / max(rank, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def save_lora_adapter(model: nn.Module, path: str) -> None:
    """Save just the LoRA adapter weights to ``path``."""
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        model.save_pretrained(path)
        return
    state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(state, path)


def load_lora_adapter(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA adapter weights from ``path`` into a wrapped model."""
    if PEFT_AVAILABLE and isinstance(model, PeftModel):
        model.load_adapter(path, adapter_name="default")
        return model
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load_lora_adapter] missing={len(missing)} unexpected={len(unexpected)}")
    return model
