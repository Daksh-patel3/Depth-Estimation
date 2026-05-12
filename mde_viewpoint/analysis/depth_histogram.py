"""Predicted vs ground-truth depth histograms per viewpoint category.

Aggregates predicted and GT depth values across a dataloader, normalizes the
two histograms, and overlays them on a single set of axes. Top-down (Category
D) views typically have a near-uniform GT distribution because every pixel
is roughly the same distance from the camera; deviations of the predicted
histogram from this expected shape are a quick visual proxy for failure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval.metrics import apply_median_scaling


def collect_depth_values(
    model,
    dataloader: DataLoader,
    n_samples: int = 200,
    is_metric: Optional[bool] = None,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gather flattened predicted and GT depth values across ``n_samples``.

    Returns ``(pred_values, gt_values)`` 1-D numpy arrays containing only
    masked, finite, in-range pixels.
    """
    if is_metric is None:
        is_metric = bool(getattr(model, "is_metric", False))

    pred_chunks: List[np.ndarray] = []
    gt_chunks: List[np.ndarray] = []
    seen = 0
    for batch in dataloader:
        image = batch["image"]
        depth = batch["depth"]
        mask = batch["mask"]
        try:
            pred = model.predict(image)
        except Exception as exc:
            print(f"[depth_histogram] predict failed: {exc}")
            continue
        pred = pred.detach().cpu()
        for i in range(pred.shape[0]):
            if seen >= n_samples:
                break
            p = pred[i, 0].numpy()
            g = depth[i, 0].numpy()
            m = mask[i, 0].numpy().astype(bool)
            if not is_metric:
                p = apply_median_scaling(p, g, m)
            valid = m & np.isfinite(p) & np.isfinite(g)
            valid &= (g >= min_depth) & (g <= max_depth)
            if not valid.any():
                continue
            pred_chunks.append(np.clip(p[valid], min_depth, max_depth))
            gt_chunks.append(np.clip(g[valid], min_depth, max_depth))
            seen += 1
        if seen >= n_samples:
            break
    if not pred_chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return np.concatenate(pred_chunks), np.concatenate(gt_chunks)


def plot_histograms(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    save_path: str,
    bins: int = 64,
    category: str = "",
    expected_uniform: bool = False,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> None:
    """Save a normalized overlay histogram of predicted vs GT depth values."""
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_edges = np.linspace(min_depth, max_depth, bins + 1)

    if gt_values.size > 0:
        ax.hist(gt_values, bins=bin_edges, density=True, alpha=0.55,
                label="GT", color="tab:blue")
    if pred_values.size > 0:
        ax.hist(pred_values, bins=bin_edges, density=True, alpha=0.55,
                label="Pred", color="tab:orange")

    if expected_uniform and gt_values.size > 0:
        # Annotate a horizontal reference line at the uniform density level.
        density = 1.0 / (max_depth - min_depth)
        ax.axhline(density, color="black", linestyle="--", alpha=0.6,
                   label="expected uniform (top-down)")

    ax.set_xlabel("depth (m)")
    ax.set_ylabel("density")
    title = "Depth distribution" + (f" — Category {category}" if category else "")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_histogram(
    model,
    dataloader: DataLoader,
    output_dir: str,
    category: str = "",
    n_samples: int = 200,
    bins: int = 64,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Compute and save the histogram plot. Returns simple summary statistics."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pred, gt = collect_depth_values(
        model, dataloader,
        n_samples=n_samples,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    save_path = os.path.join(output_dir, f"hist_category_{category or 'all'}.png")
    plot_histograms(
        pred, gt,
        save_path=save_path,
        bins=bins,
        category=category,
        expected_uniform=(category == "D"),
        min_depth=min_depth,
        max_depth=max_depth,
    )
    summary = {
        "n_pred": int(pred.size),
        "n_gt": int(gt.size),
        "pred_mean": float(pred.mean()) if pred.size else float("nan"),
        "pred_std": float(pred.std()) if pred.size else float("nan"),
        "gt_mean": float(gt.mean()) if gt.size else float("nan"),
        "gt_std": float(gt.std()) if gt.size else float("nan"),
    }
    return summary
