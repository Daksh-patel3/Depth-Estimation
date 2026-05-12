"""LoRA fine-tuning loop for Depth Anything V2.

Trains the LoRA adapters with AdamW + scale-invariant loss, validates after
each epoch, applies early stopping, and writes per-epoch metrics to CSV.
Designed to be called either from a Python script (``LoRATrainer.train``) or
via the experiment driver in ``experiments/run_lora_adaptation.py``.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ..data.dataloader import ViewpointDepthDataset, make_dataloader
from ..eval.metrics import (
    abs_rel,
    apply_median_scaling,
    delta1,
    rmse,
    scale_invariant_loss,
)
from ..models.lora_wrapper import (
    count_parameters,
    get_lora_model,
    save_lora_adapter,
)


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyperparameter container mirroring the YAML config."""

    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    max_epochs: int = 20
    early_stop_patience: int = 3
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    image_size: int = 518
    min_depth: float = 0.1
    max_depth: float = 80.0
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    grad_accum_steps: int = 1


class LoRATrainer:
    """End-to-end LoRA fine-tuning driver for Depth Anything V2.

    Parameters
    ----------
    model: nn.Module
        Either an unwrapped HF Depth Anything V2 model (we'll inject LoRA), or
        an already-LoRA-wrapped model from ``get_lora_model``.
    train_dataset, val_dataset: Dataset
        Datasets producing ``{image, depth, mask, meta}`` dicts.
    config: TrainerConfig | dict
        Hyperparameters.
    log_dir: str
        Where to write the per-epoch CSV log + best checkpoint.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        config: Any,
        log_dir: str = "./runs/lora",
        wrap_lora: bool = True,
    ):
        self.cfg = config if isinstance(config, TrainerConfig) else TrainerConfig(**{
            k: v for k, v in config.items() if k in TrainerConfig.__dataclass_fields__
        })
        seed_everything(self.cfg.seed)
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu"
        )
        if wrap_lora:
            self.model = get_lora_model(
                model,
                rank=self.cfg.lora_rank,
                alpha=self.cfg.lora_alpha,
                dropout=self.cfg.lora_dropout,
            )
        else:
            self.model = model
        self.model.to(self.device)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_dir = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("No trainable parameters — did LoRA wrap correctly?")
        self.optimizer = AdamW(
            trainable,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self._csv_path = os.path.join(self.log_dir, "training_log.csv")
        self._best_val: float = float("inf")
        self._patience: int = 0

    # ------------------------------------------------------------------
    # Forward pass adapter for the HF Depth Anything V2 model
    # ------------------------------------------------------------------
    def _forward_depth(self, image: torch.Tensor) -> torch.Tensor:
        """Run the underlying model and return [B, 1, H, W] predictions."""
        # PEFT's PeftModel.forward() injects kwargs like input_ids that
        # DepthAnything doesn't accept.  Call through base_model (the LoRA
        # tuner) which forwards directly to the HF model while still
        # applying the LoRA-modified layers.
        fwd_target = getattr(self.model, "base_model", self.model)
        try:
            out = fwd_target(pixel_values=image)
        except TypeError:
            out = fwd_target(image)
        if hasattr(out, "predicted_depth"):
            pred = out.predicted_depth
        elif isinstance(out, dict):
            pred = out.get("predicted_depth", out.get("metric_depth"))
            if pred is None:
                pred = next(iter(out.values()))
        else:
            pred = out
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if pred.shape[-2:] != image.shape[-2:]:
            pred = F.interpolate(
                pred, size=image.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        return pred.clamp(min=1e-3)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return make_dataloader(
            dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=shuffle,
            drop_last=shuffle,
        )

    def train(self, n_samples: Optional[int] = None) -> Dict[str, Any]:
        """Train on the first ``n_samples`` of the train split.

        Returns a dict with the best validation metrics and final state.
        """
        train_set = self.train_dataset
        if n_samples is not None and n_samples < len(train_set):
            indices = list(range(n_samples))
            train_set = Subset(self.train_dataset, indices)

        train_loader = self._make_loader(train_set, shuffle=True)
        val_loader = self._make_loader(self.val_dataset, shuffle=False)

        self._init_csv()
        history: List[Dict[str, float]] = []

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            elapsed = time.time() - t0

            row = {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_abs_rel": float(val_metrics["abs_rel"]),
                "val_rmse": float(val_metrics["rmse"]),
                "val_delta1": float(val_metrics["delta1"]),
                "elapsed_sec": float(elapsed),
                "n_train": len(train_set),
            }
            history.append(row)
            self._append_csv(row)
            print(
                f"[epoch {epoch:02d}] train_loss={train_loss:.4f} "
                f"abs_rel={val_metrics['abs_rel']:.4f} "
                f"rmse={val_metrics['rmse']:.4f} "
                f"delta1={val_metrics['delta1']:.4f} ({elapsed:.1f}s)"
            )

            improved = val_metrics["abs_rel"] < self._best_val - 1e-6
            if improved:
                self._best_val = float(val_metrics["abs_rel"])
                self._patience = 0
                self.save_checkpoint(os.path.join(self.log_dir, "best_adapter"))
            else:
                self._patience += 1
                if self._patience >= self.cfg.early_stop_patience:
                    print(f"[early stop] no improvement for {self._patience} epochs.")
                    break

        return {
            "best_val_abs_rel": self._best_val,
            "history": history,
            "n_train": len(train_set),
        }

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()
        for batch_idx, batch in enumerate(loader):
            image = batch["image"].to(self.device, non_blocking=True)
            depth = batch["depth"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)

            pred = self._forward_depth(image)
            loss = scale_invariant_loss(pred, depth, mask)
            loss = loss / max(self.cfg.grad_accum_steps, 1)
            loss.backward()

            if (batch_idx + 1) % self.cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += float(loss.item()) * self.cfg.grad_accum_steps
            n_batches += 1
        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Public wrapper: run validation on ``self.val_dataset``."""
        loader = self._make_loader(self.val_dataset, shuffle=False)
        return self._validate(loader)

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        absrels: List[float] = []
        rmses: List[float] = []
        deltas: List[float] = []
        for batch in loader:
            image = batch["image"].to(self.device, non_blocking=True)
            depth = batch["depth"]
            mask = batch["mask"]
            pred = self._forward_depth(image)
            pred = pred.detach().cpu()
            for i in range(pred.shape[0]):
                p = pred[i, 0].numpy()
                g = depth[i, 0].numpy()
                m = mask[i, 0].numpy().astype(bool)
                # Treat fine-tuned model as relative-depth for stability —
                # match the prediction scale to GT before computing metrics.
                p_scaled = apply_median_scaling(p, g, m)
                p_scaled = np.clip(p_scaled, self.cfg.min_depth, self.cfg.max_depth)
                absrels.append(abs_rel(p_scaled, g, m))
                rmses.append(rmse(p_scaled, g, m))
                deltas.append(delta1(p_scaled, g, m))
        return {
            "abs_rel": float(np.nanmean(absrels)) if absrels else float("nan"),
            "rmse": float(np.nanmean(rmses)) if rmses else float("nan"),
            "delta1": float(np.nanmean(deltas)) if deltas else float("nan"),
        }

    # ------------------------------------------------------------------
    # Checkpoints + logging
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str) -> None:
        """Save LoRA adapter weights to ``path`` (creates the directory)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        save_lora_adapter(self.model, path)

    def _init_csv(self) -> None:
        with open(self._csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                "epoch", "train_loss", "val_abs_rel", "val_rmse",
                "val_delta1", "elapsed_sec", "n_train",
            ])

    def _append_csv(self, row: Dict[str, float]) -> None:
        with open(self._csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([
                row["epoch"], f"{row['train_loss']:.6f}",
                f"{row['val_abs_rel']:.6f}", f"{row['val_rmse']:.6f}",
                f"{row['val_delta1']:.6f}", f"{row['elapsed_sec']:.3f}",
                row["n_train"],
            ])
