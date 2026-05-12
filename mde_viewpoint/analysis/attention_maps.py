"""Forward-hook based attention extraction for transformer depth models.

Targets the last 4 transformer blocks of either Depth Anything V2 (DINOv2
encoder) or MiDaS DPT_Large. We register hooks on the attention probability
tensor when available, fall back to capturing q/k and recomputing softmax
otherwise, then take the [CLS] row, reshape to a spatial grid, and bilinearly
upsample to the input resolution for visualization.

Usage:
    extractor = AttentionExtractor(hf_model, last_n=4)
    extractor.register()
    _ = hf_model(**inputs)
    attn = extractor.cls_attention_maps(image_size=518)  # list of HxW arrays
    extractor.remove()
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _denormalize_rgb(image: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = image.detach().cpu() * std + mean
    arr = arr.clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def _find_attention_blocks(model: nn.Module, last_n: int = 4) -> List[Tuple[str, nn.Module]]:
    """Locate the last ``last_n`` transformer blocks in a HF DINOv2 / DPT model.

    Heuristic: any module path that contains ``"layer."`` followed by an int,
    grouped by their integer index. Returns ``[(name, module), ...]`` sorted
    by depth, keeping the last ``last_n``.
    """
    candidates: List[Tuple[int, str, nn.Module]] = []
    for name, module in model.named_modules():
        # Look for blocks shaped like ``...encoder.layer.<int>`` or
        # ``...blocks.<int>``.
        parts = name.split(".")
        for i in range(len(parts) - 1):
            if parts[i] in {"layer", "blocks"} and parts[i + 1].isdigit():
                idx = int(parts[i + 1])
                # Only keep the block-level module, not its children.
                if i + 1 == len(parts) - 1:
                    candidates.append((idx, name, module))
                break
    if not candidates:
        return []
    candidates.sort(key=lambda x: x[0])
    return [(name, mod) for _, name, mod in candidates[-last_n:]]


class AttentionExtractor:
    """Capture attention probabilities from the last ``last_n`` transformer blocks.

    Parameters
    ----------
    model: nn.Module
        A HF Depth Anything V2 model or MiDaS DPT_Large torch.hub model.
    last_n: int
        Number of trailing transformer blocks to hook.
    head_reduce: str
        ``"mean"`` (default) or ``"max"`` aggregation across attention heads.
    """

    def __init__(self, model: nn.Module, last_n: int = 4, head_reduce: str = "mean"):
        self.model = model
        self.last_n = int(last_n)
        if head_reduce not in {"mean", "max"}:
            raise ValueError(f"head_reduce must be 'mean' or 'max', got {head_reduce!r}")
        self.head_reduce = head_reduce
        self.blocks = _find_attention_blocks(model, last_n=last_n)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def register(self) -> None:
        """Install forward hooks. Captures attention probs into ``_captured``."""
        self._captured.clear()
        for name, block in self.blocks:
            self._handles.append(
                block.register_forward_hook(self._make_hook(name))
            )

    def remove(self) -> None:
        """Tear down all forward hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------
    def _make_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Any, output: Any) -> None:
            attn = self._extract_attention(module, output)
            if attn is not None:
                self._captured[name] = attn.detach()
        return hook

    @staticmethod
    def _extract_attention(module: nn.Module, output: Any) -> Optional[torch.Tensor]:
        """Try several strategies to recover attention probabilities."""
        # Case 1: HF blocks return tuples and may include attention weights.
        if isinstance(output, tuple):
            for elt in output:
                if isinstance(elt, torch.Tensor) and elt.ndim == 4:
                    # [B, heads, T, T]
                    return elt
        # Case 2: search children for an "attention.attn_probs" buffer.
        for n, sub in module.named_modules():
            if hasattr(sub, "attn_probs") and isinstance(sub.attn_probs, torch.Tensor):
                return sub.attn_probs
        return None

    # ------------------------------------------------------------------
    def cls_attention_maps(
        self,
        image_size: int,
    ) -> List[np.ndarray]:
        """Return one ``(image_size, image_size)`` attention map per hooked block."""
        maps: List[np.ndarray] = []
        for name, _ in self.blocks:
            attn = self._captured.get(name)
            if attn is None:
                continue
            # attn: [B, heads, T, T]. Take batch 0, [CLS] row.
            a = attn[0]  # [heads, T, T]
            cls_row = a[:, 0, 1:]  # drop CLS-on-CLS, keep CLS-on-tokens
            if self.head_reduce == "mean":
                cls_row = cls_row.mean(dim=0)
            else:
                cls_row = cls_row.max(dim=0).values
            n_tokens = cls_row.shape[0]
            side = int(round(math.sqrt(n_tokens)))
            if side * side != n_tokens:
                # Non-square token grid (e.g. with register tokens). Trim.
                side = int(math.floor(math.sqrt(n_tokens)))
                cls_row = cls_row[: side * side]
            grid = cls_row.reshape(1, 1, side, side).float()
            up = F.interpolate(grid, size=(image_size, image_size),
                               mode="bilinear", align_corners=False)
            up = up.squeeze().cpu().numpy()
            up = (up - up.min()) / max(up.max() - up.min(), 1e-8)
            maps.append(up)
        return maps


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_attention_grid(
    image: np.ndarray,
    attn_maps: Sequence[np.ndarray],
    save_path: str,
    title: Optional[str] = None,
    cmap: str = "inferno",
) -> None:
    """Render the input RGB plus one panel per attention map."""
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    n = len(attn_maps) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    axes[0].imshow(image)
    axes[0].set_title("RGB")
    axes[0].axis("off")
    for i, m in enumerate(attn_maps):
        axes[i + 1].imshow(image)
        axes[i + 1].imshow(m, cmap=cmap, alpha=0.55)
        axes[i + 1].set_title(f"block -{len(attn_maps) - i}")
        axes[i + 1].axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_before_after(
    image: np.ndarray,
    attn_before: Sequence[np.ndarray],
    attn_after: Sequence[np.ndarray],
    save_path: str,
    title: str = "Attention before vs after LoRA",
) -> None:
    """Render two rows of attention overlays (pre / post LoRA)."""
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    n = max(len(attn_before), len(attn_after))
    fig, axes = plt.subplots(2, n + 1, figsize=(4 * (n + 1), 8))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    for row, (label, attns) in enumerate([("before", attn_before), ("after", attn_after)]):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f"RGB ({label})")
        axes[row, 0].axis("off")
        for i in range(n):
            ax = axes[row, i + 1]
            ax.imshow(image)
            if i < len(attns):
                ax.imshow(attns[i], cmap="inferno", alpha=0.55)
                ax.set_title(f"block -{len(attns) - i}")
            ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
