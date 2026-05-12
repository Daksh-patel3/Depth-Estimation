"""Per-pixel error heatmaps and aggregate spatial-bias maps.

For a given (image, prediction, ground-truth) triple, computes the absolute
relative error per pixel, optionally overlays it on the RGB input, and saves
a figure to disk. Also supports aggregating errors over a dataset onto a
coarse 16x16 grid to expose systematic spatial biases for a viewpoint
category (e.g. top-down predictions consistently failing in the image
center).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..eval.metrics import apply_median_scaling


def _denormalize_rgb(image: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for visualization. Returns HxWx3 uint8."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    arr = image.detach().cpu() * std + mean
    arr = arr.clamp(0, 1).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def per_pixel_abs_rel(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return |pred - gt| / gt with NaNs where mask is False or gt invalid."""
    err = np.full_like(gt, fill_value=np.nan, dtype=np.float32)
    valid = mask.astype(bool) & np.isfinite(gt) & (gt > 0) & np.isfinite(pred)
    err[valid] = np.abs(pred[valid] - gt[valid]) / np.clip(gt[valid], 1e-6, None)
    return err


def render_error_heatmap(
    image: np.ndarray,
    error_map: np.ndarray,
    save_path: str,
    title: Optional[str] = None,
    vmax: float = 1.0,
) -> None:
    """Save a 3-panel figure: RGB | error heatmap | overlay."""
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("RGB")
    axes[0].axis("off")

    em = np.where(np.isfinite(error_map), error_map, 0.0)
    axes[1].imshow(em, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title("|pred - gt| / gt")
    axes[1].axis("off")

    axes[2].imshow(image)
    axes[2].imshow(em, cmap="viridis", vmin=0, vmax=vmax, alpha=0.55)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def aggregate_spatial_bias(
    error_maps: Sequence[np.ndarray],
    grid: int = 16,
) -> np.ndarray:
    """Bin per-pixel errors onto a coarse ``grid x grid`` cell grid.

    Returns a ``grid x grid`` array of mean errors, ignoring NaN cells. Useful
    for showing that, e.g., top-down depth predictions degrade more strongly
    in the image center than at the borders.
    """
    if not error_maps:
        return np.zeros((grid, grid), dtype=np.float32)
    h, w = error_maps[0].shape
    cell_h = max(h // grid, 1)
    cell_w = max(w // grid, 1)
    sums = np.zeros((grid, grid), dtype=np.float64)
    counts = np.zeros((grid, grid), dtype=np.int64)
    for em in error_maps:
        if em.shape != (h, w):
            continue
        for r in range(grid):
            for c in range(grid):
                tile = em[r * cell_h: (r + 1) * cell_h, c * cell_w: (c + 1) * cell_w]
                valid = np.isfinite(tile)
                if valid.any():
                    sums[r, c] += float(tile[valid].sum())
                    counts[r, c] += int(valid.sum())
    out = np.full((grid, grid), np.nan, dtype=np.float32)
    nonzero = counts > 0
    out[nonzero] = (sums[nonzero] / counts[nonzero]).astype(np.float32)
    return out


def render_spatial_bias(
    bias: np.ndarray,
    save_path: str,
    title: str = "Mean |pred - gt| / gt by image cell",
    vmax: Optional[float] = None,
) -> None:
    """Save a heatmap of the aggregate spatial-bias array."""
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(bias, cmap="viridis", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("grid column")
    ax.set_ylabel("grid row")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def run_error_analysis(
    model,
    dataloader: DataLoader,
    output_dir: str,
    n_images: int = 20,
    grid: int = 16,
    is_metric: Optional[bool] = None,
) -> np.ndarray:
    """Render per-image heatmaps for the first ``n_images`` and the bias grid.

    Returns the aggregated spatial-bias array.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if is_metric is None:
        is_metric = bool(getattr(model, "is_metric", False))

    all_error_maps: List[np.ndarray] = []
    rendered = 0
    for batch in dataloader:
        image = batch["image"]
        depth = batch["depth"]
        mask = batch["mask"]
        try:
            pred = model.predict(image)
        except Exception as exc:
            print(f"[error_heatmap] predict failed: {exc}")
            continue
        pred = pred.detach().cpu()
        for i in range(pred.shape[0]):
            p = pred[i, 0].numpy()
            g = depth[i, 0].numpy()
            m = mask[i, 0].numpy().astype(bool)
            if not is_metric:
                p = apply_median_scaling(p, g, m)
            em = per_pixel_abs_rel(p, g, m)
            all_error_maps.append(em)
            if rendered < n_images:
                rgb = _denormalize_rgb(image[i])
                save_path = os.path.join(output_dir, f"err_{rendered:03d}.png")
                render_error_heatmap(rgb, em, save_path,
                                     title=f"sample {rendered}")
                rendered += 1
        if rendered >= n_images and len(all_error_maps) >= n_images * 2:
            break

    bias = aggregate_spatial_bias(all_error_maps, grid=grid)
    render_spatial_bias(bias, os.path.join(output_dir, "spatial_bias.png"))
    np.save(os.path.join(output_dir, "spatial_bias.npy"), bias)
    return bias
