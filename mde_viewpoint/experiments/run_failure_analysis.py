"""Step 3: failure-mode analysis (heatmaps, attention, histograms).

For a chosen ``--model`` and ``--category``, runs the three analysis modules
and writes figures + JSON summaries under ``results/failure_analysis/``.

Usage:
    python -m mde_viewpoint.experiments.run_failure_analysis \
        --model dav2 --category D --n_images 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

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

import torch

from mde_viewpoint.analysis.attention_maps import (
    AttentionExtractor,
    render_attention_grid,
    _denormalize_rgb as _attn_denormalize_rgb,
)
from mde_viewpoint.analysis.depth_histogram import run_histogram
from mde_viewpoint.analysis.error_heatmap import run_error_analysis
from mde_viewpoint.data.dataloader import load_category_dataloader
from mde_viewpoint.models.model_zoo import build_model


# ---------------------------------------------------------------------------
# Attention helper
# ---------------------------------------------------------------------------

def run_attention(model, dataloader, output_dir: str, n_images: int = 8) -> int:
    """Render attention overlays for the first ``n_images`` images in the loader.

    Only meaningful for transformer-backed models (Depth Anything V2, MiDaS
    DPT_Large). For non-transformer or pipeline models (Marigold) we skip.
    """
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "named_modules"):
        print("[attention] skipping (model has no named_modules).")
        return 0

    extractor = AttentionExtractor(base, last_n=4)
    if not extractor.blocks:
        print("[attention] no attention blocks found, skipping.")
        return 0

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    extractor.register()
    rendered = 0
    try:
        for batch in dataloader:
            image = batch["image"]
            try:
                _ = model.predict(image)
            except Exception as exc:
                print(f"[attention] forward failed: {exc}")
                continue
            for i in range(image.shape[0]):
                if rendered >= n_images:
                    break
                rgb = _attn_denormalize_rgb(image[i])
                maps = extractor.cls_attention_maps(image_size=image.shape[-1])
                if not maps:
                    continue
                save_path = os.path.join(output_dir, f"attn_{rendered:03d}.png")
                render_attention_grid(rgb, maps, save_path,
                                      title=f"Attention sample {rendered}")
                rendered += 1
            if rendered >= n_images:
                break
    finally:
        extractor.remove()
    return rendered


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-category failure-mode analysis.")
    parser.add_argument("--model", required=True,
                        choices=["dav2", "zoedepth", "midas", "marigold"])
    parser.add_argument("--category", required=True,
                        choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--n_images", type=int, default=50)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))

    output_root = args.output_root or os.path.join(
        config["results_dir"], "failure_analysis", args.model, args.category,
    )
    Path(output_root).mkdir(parents=True, exist_ok=True)

    loader = load_category_dataloader(
        split_dir=config["split_dir"],
        category=args.category,
        subset="eval",
        image_size=int(config["image_size"]),
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )

    model = build_model(args.model, device=config["device"])

    # 1) Per-pixel error heatmaps + spatial bias.
    heat_dir = os.path.join(output_root, "heatmaps")
    bias = run_error_analysis(model, loader, heat_dir, n_images=args.n_images)

    # 2) Attention maps (transformer models only).
    attn_dir = os.path.join(output_root, "attention")
    n_attn = run_attention(model, loader, attn_dir, n_images=min(8, args.n_images))

    # 3) Depth histogram.
    hist_dir = os.path.join(output_root, "histograms")
    hist_summary = run_histogram(
        model, loader, hist_dir,
        category=args.category,
        n_samples=args.n_images,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
    )

    summary: Dict[str, Any] = {
        "model": args.model,
        "category": args.category,
        "n_images": args.n_images,
        "spatial_bias_min": float(bias[~(bias != bias)].min()) if bias.size else None,
        "spatial_bias_max": float(bias[~(bias != bias)].max()) if bias.size else None,
        "attention_rendered": n_attn,
        "histogram": hist_summary,
    }
    with open(os.path.join(output_root, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\n=== Failure analysis complete ===")
    print(json.dumps(summary, indent=2, default=float))
    print(f"\nWrote {output_root}")


if __name__ == "__main__":
    main()
