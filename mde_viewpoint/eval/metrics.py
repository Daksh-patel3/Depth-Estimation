"""Depth-estimation metrics and training losses.

All numpy-based metric functions accept ``(pred, gt, mask)`` arrays and return
a Python float. Inputs are expected to be HxW or 1xHxW float arrays in meters
for metric depth, or arbitrary positive units for relative depth (use
``median_scale`` first in that case).

The training loss ``scale_invariant_loss`` operates on torch tensors and is
differentiable.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Normalize input to a 2D float32 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr.squeeze(0)
    return arr


def _coerce(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert pred / gt / mask to consistent 2D numpy arrays."""
    p = _to_numpy(pred)
    g = _to_numpy(gt)
    m = _to_numpy(mask).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"pred shape {p.shape} != gt shape {g.shape}")
    if m.shape != g.shape:
        raise ValueError(f"mask shape {m.shape} != gt shape {g.shape}")
    # Drop non-finite predictions / gt from the mask.
    finite = np.isfinite(p) & np.isfinite(g) & (g > 0)
    m = m & finite
    return p, g, m


def median_scale(pred: ArrayLike, gt: ArrayLike, mask: ArrayLike) -> float:
    """Return the median ratio that scales ``pred`` to match ``gt`` (relative-depth models)."""
    p, g, m = _coerce(pred, gt, mask)
    if m.sum() == 0:
        return 1.0
    p_med = float(np.median(p[m]))
    g_med = float(np.median(g[m]))
    if p_med <= 1e-8:
        return 1.0
    return g_med / p_med


def apply_median_scaling(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: ArrayLike,
) -> np.ndarray:
    """Return a scaled copy of ``pred`` so that median(pred*s) == median(gt) on mask."""
    s = median_scale(pred, gt, mask)
    return _to_numpy(pred) * s


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def abs_rel(pred: ArrayLike, gt: ArrayLike, mask: ArrayLike) -> float:
    """Mean absolute relative error: mean(|pred - gt| / gt) on the mask."""
    p, g, m = _coerce(pred, gt, mask)
    if m.sum() == 0:
        return float("nan")
    diff = np.abs(p[m] - g[m]) / np.clip(g[m], 1e-6, None)
    return float(np.mean(diff))


def rmse(pred: ArrayLike, gt: ArrayLike, mask: ArrayLike) -> float:
    """Root mean squared error on the mask."""
    p, g, m = _coerce(pred, gt, mask)
    if m.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p[m] - g[m]) ** 2)))


def delta1(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: ArrayLike,
    threshold: float = 1.25,
) -> float:
    """Fraction of pixels where max(pred/gt, gt/pred) < threshold."""
    p, g, m = _coerce(pred, gt, mask)
    if m.sum() == 0:
        return float("nan")
    p_v = np.clip(p[m], 1e-6, None)
    g_v = np.clip(g[m], 1e-6, None)
    ratio = np.maximum(p_v / g_v, g_v / p_v)
    return float(np.mean(ratio < threshold))


def degradation_ratio(metric_extreme: float, metric_eye_level: float) -> float:
    """Relative degradation: (extreme / eye_level) for error-style metrics.

    For metrics where lower is better (AbsRel, RMSE) a value > 1 means worse.
    For metrics where higher is better (delta1) the caller should invert the
    ratio (eye_level / extreme) before passing in.
    """
    if metric_eye_level == 0 or not np.isfinite(metric_eye_level):
        return float("nan")
    return float(metric_extreme / metric_eye_level)


# ---------------------------------------------------------------------------
# Combined evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    pred: ArrayLike,
    gt: ArrayLike,
    mask: ArrayLike,
    is_metric: bool,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> dict:
    """Compute the standard metric triplet for a single (pred, gt) pair.

    If ``is_metric`` is False, applies median scaling first. ``min_depth`` and
    ``max_depth`` clip the prediction prior to evaluation.
    """
    pred_np = _to_numpy(pred)
    gt_np = _to_numpy(gt)
    mask_np = _to_numpy(mask).astype(bool)

    if not is_metric:
        pred_np = apply_median_scaling(pred_np, gt_np, mask_np)

    pred_np = np.clip(pred_np, min_depth, max_depth)
    return {
        "abs_rel": abs_rel(pred_np, gt_np, mask_np),
        "rmse": rmse(pred_np, gt_np, mask_np),
        "delta1": delta1(pred_np, gt_np, mask_np),
        "n_valid": int(mask_np.sum()),
    }


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------

def scale_invariant_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.85,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Scale-invariant log loss (Eigen et al., 2014).

    L = mean(d^2) - lam * (mean(d))^2  where d = log(pred) - log(gt)
    on the masked pixels. Returns a scalar tensor with grad enabled.
    """
    if pred.shape != gt.shape:
        # Resize prediction to gt shape if necessary.
        pred = torch.nn.functional.interpolate(
            pred, size=gt.shape[-2:], mode="bilinear", align_corners=False
        )
    if mask.shape != gt.shape:
        mask = torch.nn.functional.interpolate(
            mask.float(), size=gt.shape[-2:], mode="nearest"
        )
    mask_b = mask > 0
    if mask_b.sum() == 0:
        return pred.sum() * 0.0  # zero loss with grad path

    pred_log = torch.log(pred.clamp_min(eps))
    gt_log = torch.log(gt.clamp_min(eps))
    diff = (pred_log - gt_log)[mask_b]
    n = diff.numel()
    term1 = (diff ** 2).mean()
    term2 = (diff.sum() / n) ** 2
    loss = term1 - lam * term2
    return loss
