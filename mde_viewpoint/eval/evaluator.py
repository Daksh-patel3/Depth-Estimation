"""Generic depth-estimation evaluator.

Loops a ``DepthModel`` over a ``DataLoader``, computes per-sample metrics, and
aggregates them into a single dict. Designed to be reused by every script in
``experiments/`` (zero-shot eval, fine-tuned eval, NYU/KITTI eval).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import abs_rel, apply_median_scaling, delta1, evaluate_pair, rmse


class Evaluator:
    """Run a depth model over a dataloader and aggregate per-sample metrics.

    Parameters
    ----------
    model: DepthModel
        Any wrapper from ``models.model_zoo`` (must expose ``predict`` and
        ``is_metric``).
    dataloader: DataLoader
        Yields dicts with ``image``, ``depth``, ``mask`` keys (see
        ``data.dataloader.ViewpointDepthDataset``).
    metrics: Sequence[str]
        Subset of ``{"abs_rel", "rmse", "delta1"}`` to track.
    min_depth, max_depth: float
        Clip predictions to this range before evaluation.
    device: str
        Device to pin tensors on.
    """

    SUPPORTED_METRICS = ("abs_rel", "rmse", "delta1")

    def __init__(
        self,
        model: Any,
        dataloader: DataLoader,
        metrics: Sequence[str] = ("abs_rel", "rmse", "delta1"),
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        device: str = "cuda",
    ):
        self.model = model
        self.dataloader = dataloader
        self.metrics = tuple(m for m in metrics if m in self.SUPPORTED_METRICS)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._results: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def _predict_batch(self, image: torch.Tensor) -> torch.Tensor:
        image = image.to(self.device, non_blocking=True)
        if hasattr(self.model, "predict"):
            return self.model.predict(image)
        # Fallback: assume the model is a torch.nn.Module that returns depth.
        with torch.no_grad():
            out = self.model(image)
        if isinstance(out, dict):
            out = out.get("predicted_depth", next(iter(out.values())))
        if out.ndim == 3:
            out = out.unsqueeze(1)
        return out

    @property
    def is_metric(self) -> bool:
        return bool(getattr(self.model, "is_metric", False))

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, max_batches: Optional[int] = None) -> Dict[str, Any]:
        """Run evaluation. ``max_batches`` truncates for quick smoke tests."""
        per_sample: List[Dict[str, float]] = []
        category_counts: Dict[str, int] = {}
        t0 = time.time()
        try:
            self.model.eval()
        except Exception:
            pass

        for batch_idx, batch in enumerate(tqdm(self.dataloader, desc="eval")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            image = batch["image"]
            depth = batch["depth"]
            mask = batch["mask"]
            metas = batch.get("meta", {})
            try:
                pred = self._predict_batch(image)
            except Exception as exc:
                print(f"[evaluator] predict failed on batch {batch_idx}: {exc}")
                continue

            pred = pred.detach().cpu()
            depth = depth.detach().cpu()
            mask = mask.detach().cpu()

            for i in range(pred.shape[0]):
                p_i = pred[i]
                g_i = depth[i]
                m_i = mask[i]
                metrics_i = evaluate_pair(
                    p_i, g_i, m_i,
                    is_metric=self.is_metric,
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                )
                category = ""
                if isinstance(metas, dict) and "category" in metas:
                    cat = metas["category"]
                    category = cat[i] if isinstance(cat, (list, tuple)) else str(cat)
                metrics_i["category"] = category
                per_sample.append(metrics_i)
                category_counts[category] = category_counts.get(category, 0) + 1

        elapsed = time.time() - t0
        agg = self._aggregate(per_sample)
        agg["n_samples"] = len(per_sample)
        agg["elapsed_sec"] = elapsed
        agg["category_counts"] = category_counts
        agg["per_sample"] = per_sample
        self._results = agg
        return {k: v for k, v in agg.items() if k != "per_sample"}

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------
    def _aggregate(self, per_sample: Sequence[Dict[str, float]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for m in self.metrics:
            vals = [s[m] for s in per_sample if m in s and np.isfinite(s.get(m, np.nan))]
            out[m] = float(np.mean(vals)) if vals else float("nan")
            out[f"{m}_std"] = float(np.std(vals)) if vals else float("nan")
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_results(self, path: str, include_per_sample: bool = False) -> None:
        """Save aggregated metrics (and optionally per-sample) as JSON."""
        if self._results is None:
            raise RuntimeError("Call .run() before .save_results().")
        out = dict(self._results)
        if not include_per_sample:
            out.pop("per_sample", None)
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print(f"[evaluator] wrote {path}")
