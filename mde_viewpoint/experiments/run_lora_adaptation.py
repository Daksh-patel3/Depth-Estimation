"""Step 4: LoRA adaptation curves on a target viewpoint category.

Fine-tunes Depth Anything V2 with LoRA on N samples for N in {50, 100, 200,
500} (configurable), evaluates after each run, and saves the adaptation
curve (N vs AbsRel/RMSE/delta1) as both CSV and PNG.

Usage:
    python -m mde_viewpoint.experiments.run_lora_adaptation \
        --category D --rank 8
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

if __name__ == "__main__" and __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mde_viewpoint.experiments._common import (
        ensure_project_root_on_path,
        load_config,
        seed_everything,
    )
else:
    from ._common import ensure_project_root_on_path, load_config, seed_everything

ensure_project_root_on_path()

import matplotlib.pyplot as plt
import torch

from mde_viewpoint.data.dataloader import ViewpointDepthDataset
from mde_viewpoint.train.lora_finetune import LoRATrainer, TrainerConfig


def _build_datasets(split_path: str, image_size: int,
                    min_depth: float, max_depth: float):
    """Build train / val datasets from a per-category JSON split."""
    train = ViewpointDepthDataset.from_split(
        split_path, subset="finetune",
        image_size=image_size, normalize=True,
        min_depth=min_depth, max_depth=max_depth,
        augment=True,
    )
    val = ViewpointDepthDataset.from_split(
        split_path, subset="eval",
        image_size=image_size, normalize=True,
        min_depth=min_depth, max_depth=max_depth,
        augment=False,
    )
    return train, val


def _load_dav2():
    """Load the unwrapped HF Depth Anything V2 base model."""
    from transformers import AutoModelForDepthEstimation
    return AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Large-hf"
    )


def plot_curve(
    rows: List[Dict[str, float]],
    save_path: str,
    title: str,
) -> None:
    """Plot N (x-axis) vs three metrics (twin axes for AbsRel/RMSE vs delta1)."""
    Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
    n = [r["n_samples"] for r in rows]
    abs_rel = [r["abs_rel"] for r in rows]
    rmse = [r["rmse"] for r in rows]
    d1 = [r["delta1"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(n, abs_rel, marker="o", color="tab:red", label="AbsRel ↓")
    ax1.plot(n, rmse, marker="s", color="tab:purple", label="RMSE ↓")
    ax1.set_xlabel("# fine-tune samples (N)")
    ax1.set_ylabel("AbsRel / RMSE (lower is better)")
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    ax2.plot(n, d1, marker="^", color="tab:green", label="δ<1.25 ↑")
    ax2.set_ylabel("δ<1.25 (higher is better)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA adaptation curve experiment.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--category", required=True,
                        choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--rank", type=int, default=8, choices=[8, 16])
    parser.add_argument("--n_list", nargs="*", type=int, default=None)
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    n_list = args.n_list or config["n_finetune_samples"]
    split_path = os.path.join(config["split_dir"], f"category_{args.category}.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    train_ds, val_ds = _build_datasets(
        split_path,
        image_size=int(config["image_size"]),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )
    output_root = args.output_root or os.path.join(
        config["results_dir"], "lora", f"category_{args.category}_rank{args.rank}",
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    csv_path = os.path.join(output_root, "adaptation_curve.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n_samples", "abs_rel", "rmse", "delta1", "best_val_abs_rel"])

    for n in n_list:
        run_dir = os.path.join(output_root, f"N_{n}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        print(f"\n=== LoRA adaptation: category={args.category} rank={args.rank} N={n} ===")
        seed_everything(int(config.get("seed", 42)))

        base = _load_dav2()
        trainer_cfg = TrainerConfig(
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
            batch_size=int(config["batch_size"]),
            max_epochs=int(config["max_epochs"]),
            early_stop_patience=int(config["early_stop_patience"]),
            lora_rank=int(args.rank),
            lora_alpha=int(config["lora_alpha"]),
            lora_dropout=float(config["lora_dropout"]),
            image_size=int(config["image_size"]),
            min_depth=float(config["min_depth"]),
            max_depth=float(config["max_depth"]),
            seed=int(config.get("seed", 42)),
            device=config["device"],
            num_workers=int(config["num_workers"]),
        )
        trainer = LoRATrainer(
            model=base,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=trainer_cfg,
            log_dir=run_dir,
        )
        train_result = trainer.train(n_samples=n)
        val_metrics = trainer.evaluate()
        row = {
            "n_samples": n,
            "abs_rel": float(val_metrics["abs_rel"]),
            "rmse": float(val_metrics["rmse"]),
            "delta1": float(val_metrics["delta1"]),
            "best_val_abs_rel": float(train_result["best_val_abs_rel"]),
        }
        rows.append(row)
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow([row[k] for k in
                             ("n_samples", "abs_rel", "rmse", "delta1", "best_val_abs_rel")])

        # Free GPU memory for the next N.
        del base, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    plot_curve(
        rows,
        save_path=os.path.join(output_root, "adaptation_curve.png"),
        title=f"LoRA adaptation — category {args.category} (rank {args.rank})",
    )
    with open(os.path.join(output_root, "adaptation_curve.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {output_root}/adaptation_curve.png")


if __name__ == "__main__":
    main()
