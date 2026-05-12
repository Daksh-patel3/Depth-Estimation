"""Step 2: zero-shot evaluation across all 4 models x 5 viewpoint categories.

Outputs:
    results/zero_shot/<model>/<category>.json   per-run JSON metrics
    results/zero_shot/zero_shot_table.csv       wide table (rows=models,
                                                cols=category x metric)
    results/zero_shot/zero_shot_table.tex       LaTeX-formatted table

Usage:
    python -m mde_viewpoint.experiments.run_zero_shot_eval \
        --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running as ``python experiments/run_zero_shot_eval.py``.
if __name__ == "__main__" and __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from mde_viewpoint.experiments._common import (
        ensure_project_root_on_path,
        load_config,
        seed_everything,
    )
else:
    from ._common import (
        ensure_project_root_on_path,
        load_config,
        seed_everything,
    )

ensure_project_root_on_path()

import numpy as np
import pandas as pd

from mde_viewpoint.data.dataloader import load_category_dataloader
from mde_viewpoint.eval.evaluator import Evaluator
from mde_viewpoint.eval.metrics import degradation_ratio
from mde_viewpoint.models.model_zoo import build_model


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_one(
    model_name: str,
    category: str,
    config: Dict[str, Any],
    output_dir: str,
    max_batches: int = None,
) -> Dict[str, Any]:
    """Evaluate a single (model, category) pair and dump JSON to disk."""
    print(f"\n=== {model_name} | category {category} ===")
    seed_everything(int(config.get("seed", 42)))

    loader = load_category_dataloader(
        split_dir=config["split_dir"],
        category=category,
        subset="eval",
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )
    if len(loader.dataset) == 0:
        print(f"[skip] empty split for category {category}")
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}

    model = build_model(model_name, device=config["device"])
    evaluator = Evaluator(
        model=model,
        dataloader=loader,
        metrics=("abs_rel", "rmse", "delta1"),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        device=config["device"],
    )
    results = evaluator.run(max_batches=max_batches)
    out_path = os.path.join(output_dir, model_name, f"{category}.json")
    evaluator.save_results(out_path, include_per_sample=False)

    # Free GPU memory before the next model.
    del model, evaluator
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def build_table(
    rows: Dict[str, Dict[str, Dict[str, float]]],
    categories: List[str],
) -> pd.DataFrame:
    """Reshape ``rows[model][cat]={abs_rel,rmse,delta1}`` into a wide DataFrame."""
    columns = []
    data: Dict[str, List[float]] = {}
    for cat in categories:
        for metric in ("abs_rel", "rmse", "delta1"):
            col = f"{cat}/{metric}"
            columns.append(col)
            data[col] = []
    index = []
    for model_name, by_cat in rows.items():
        index.append(model_name)
        for cat in categories:
            cell = by_cat.get(cat, {})
            for metric in ("abs_rel", "rmse", "delta1"):
                data[f"{cat}/{metric}"].append(float(cell.get(metric, np.nan)))
    return pd.DataFrame(data, index=index)[columns]


def add_degradation(table: pd.DataFrame, categories: List[str], baseline: str = "A") -> pd.DataFrame:
    """Add per-model degradation-ratio columns (cat/abs_rel / A/abs_rel)."""
    if baseline not in categories:
        return table
    out = table.copy()
    for cat in categories:
        if cat == baseline:
            continue
        for metric in ("abs_rel", "rmse"):
            base_col = f"{baseline}/{metric}"
            cat_col = f"{cat}/{metric}"
            if base_col in out.columns and cat_col in out.columns:
                out[f"{cat}/{metric}_degr"] = out.apply(
                    lambda r: degradation_ratio(r[cat_col], r[base_col]), axis=1,
                )
    return out


def to_latex(df: pd.DataFrame) -> str:
    """Render the wide table to a LaTeX string with sensible formatting."""
    return df.to_latex(float_format=lambda x: f"{x:.3f}" if np.isfinite(x) else "--",
                       caption="Zero-shot depth metrics across viewpoint categories.",
                       label="tab:zero_shot")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot benchmark of MDE models.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Override config.models; e.g. --models dav2 zoedepth")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Override config.categories.")
    parser.add_argument("--max_batches", type=int, default=None,
                        help="Cap eval batches per (model,category) for smoke tests.")
    parser.add_argument("--output_dir", default=None,
                        help="Override results_dir/zero_shot.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    models = args.models or config["models"]
    categories = args.categories or config["categories"]
    output_dir = args.output_dir or os.path.join(config["results_dir"], "zero_shot")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    rows: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m in models:
        rows[m] = {}
        for c in categories:
            try:
                rows[m][c] = run_one(m, c, config, output_dir,
                                     max_batches=args.max_batches)
            except Exception as exc:
                print(f"[error] model={m} category={c}: {exc}")
                rows[m][c] = {"abs_rel": float("nan"),
                              "rmse": float("nan"),
                              "delta1": float("nan")}

    table = build_table(rows, categories)
    table = add_degradation(table, categories, baseline="A")
    table_csv = os.path.join(output_dir, "zero_shot_table.csv")
    table_tex = os.path.join(output_dir, "zero_shot_table.tex")
    table.to_csv(table_csv)
    with open(table_tex, "w", encoding="utf-8") as fh:
        fh.write(to_latex(table))
    with open(os.path.join(output_dir, "zero_shot_raw.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, default=float)

    print("\n=== Zero-shot benchmark complete ===")
    print(table.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {table_csv}")
    print(f"Wrote {table_tex}")


if __name__ == "__main__":
    main()
