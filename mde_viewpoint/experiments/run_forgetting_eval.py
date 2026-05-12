"""Step 5: forgetting evaluation on NYU Depth V2 and KITTI.

Loads the LoRA-adapted Depth Anything V2 + adapter weights and the unwrapped
base model, evaluates both on NYU and KITTI eval splits, and reports the
delta in AbsRel / RMSE / delta1 (negative = forgetting on absolute-error
metrics, positive = forgetting on delta1).

Expected NYU layout (test split, official):
    <nyu_root>/
        nyu_test_rgb/<id>.png    (or .jpg)
        nyu_test_depth/<id>.png  (16-bit, divide by 1000.0)

Expected KITTI eval layout (Eigen split):
    <kitti_root>/
        kitti_eval_rgb/<id>.png
        kitti_eval_depth/<id>.png  (KITTI 16-bit, divide by 256.0)

Index files are JSON lists of ``{"image_path": ..., "depth_path": ...}``
dicts produced by a ``--build_index`` helper option below.
"""

from __future__ import annotations

import argparse
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

import numpy as np
import torch
import torch.nn.functional as F

from mde_viewpoint.data.dataloader import (
    ViewpointDepthDataset,
    make_dataloader,
)
from mde_viewpoint.eval.evaluator import Evaluator
from mde_viewpoint.models.lora_wrapper import (
    PEFT_AVAILABLE,
    get_lora_model,
    load_lora_adapter,
)
from mde_viewpoint.models.model_zoo import (
    DepthAnythingV2Model,
    _denormalize_imagenet,
    _to_pil_batch,
)


# ---------------------------------------------------------------------------
# Index building (NYU / KITTI)
# ---------------------------------------------------------------------------

def _index_nyu(root: str) -> List[Dict[str, str]]:
    """Build a list of {image_path, depth_path} dicts for NYU Depth V2 test."""
    rgb_dir = Path(root) / "nyu_test_rgb"
    dep_dir = Path(root) / "nyu_test_depth"
    samples: List[Dict[str, str]] = []
    if not (rgb_dir.is_dir() and dep_dir.is_dir()):
        return samples
    for img in sorted(rgb_dir.glob("*.[jp][pn]g")):
        depth = dep_dir / (img.stem + ".png")
        if depth.exists():
            samples.append({
                "image_path": str(img),
                "depth_path": str(depth),
                "pitch_deg": 0.0,
                "category": "NYU",
                "source": "nyu",
                "env_tag": "standard",
            })
    return samples


def _index_kitti(root: str) -> List[Dict[str, str]]:
    """Build a list of {image_path, depth_path} dicts for KITTI eval."""
    rgb_dir = Path(root) / "kitti_eval_rgb"
    dep_dir = Path(root) / "kitti_eval_depth"
    samples: List[Dict[str, str]] = []
    if not (rgb_dir.is_dir() and dep_dir.is_dir()):
        return samples
    for img in sorted(rgb_dir.glob("*.png")):
        depth = dep_dir / img.name
        if depth.exists():
            samples.append({
                "image_path": str(img),
                "depth_path": str(depth),
                "pitch_deg": 0.0,
                "category": "KITTI",
                "source": "kitti",
                "env_tag": "standard",
            })
    return samples


# ---------------------------------------------------------------------------
# Adapter helper (wraps DepthAnythingV2Model in LoRA + loads weights)
# ---------------------------------------------------------------------------

class DAV2WithLoRA(DepthAnythingV2Model):
    """Depth Anything V2 wrapper that injects + loads a LoRA adapter."""

    def __init__(self, adapter_path: str, rank: int = 8, alpha: int = 16,
                 dropout: float = 0.1, device: str = "cuda"):
        self.adapter_path = adapter_path
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        super().__init__(device=device)

    def _load_model(self) -> None:
        super()._load_model()
        self.model = get_lora_model(
            self.model, rank=self.rank,
            alpha=self.alpha, dropout=self.dropout,
            verbose=False,
        )
        if os.path.exists(self.adapter_path):
            self.model = load_lora_adapter(self.model, self.adapter_path)
            print(f"[forgetting] loaded LoRA adapter from {self.adapter_path}")
        else:
            print(f"[forgetting] adapter path not found: {self.adapter_path} "
                  "(evaluating with random LoRA — for sanity only)")
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict with LoRA-adapted model, bypassing PEFT's forward wrapper."""
        b, _, h, w = image.shape
        rgb = _denormalize_imagenet(image)
        inputs = self.processor(
            images=_to_pil_batch(rgb), return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        fwd_target = getattr(self.model, "base_model", self.model)
        out = fwd_target(**inputs)
        pred = out.predicted_depth
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        pred = pred.max() - pred + 1e-6
        return pred


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _evaluate(model, samples: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, float]:
    """Run the unified evaluator over a sample list."""
    if not samples:
        return {"abs_rel": float("nan"), "rmse": float("nan"), "delta1": float("nan")}
    ds = ViewpointDepthDataset(
        samples,
        image_size=int(config["image_size"]),
        normalize=True,
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        augment=False,
    )
    loader = make_dataloader(
        ds,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=False,
    )
    evaluator = Evaluator(
        model=model,
        dataloader=loader,
        metrics=("abs_rel", "rmse", "delta1"),
        min_depth=float(config["min_depth"]),
        max_depth=float(config["max_depth"]),
        device=config["device"],
    )
    return evaluator.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forgetting evaluation on NYU + KITTI.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--adapter_path", required=True,
                        help="Directory containing the saved LoRA adapter (best_adapter/).")
    parser.add_argument("--rank", type=int, default=8, choices=[8, 16])
    parser.add_argument("--datasets", nargs="*", default=["nyu", "kitti"],
                        choices=["nyu", "kitti"])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    output_dir = args.output_dir or os.path.join(config["results_dir"], "forgetting")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build per-dataset sample indexes.
    samples_by_ds: Dict[str, List[Dict[str, Any]]] = {}
    if "nyu" in args.datasets:
        samples_by_ds["nyu"] = _index_nyu(config["nyu_root"])
        print(f"[nyu] indexed {len(samples_by_ds['nyu'])} samples")
    if "kitti" in args.datasets:
        samples_by_ds["kitti"] = _index_kitti(config["kitti_root"])
        print(f"[kitti] indexed {len(samples_by_ds['kitti'])} samples")

    results: Dict[str, Dict[str, Dict[str, float]]] = {"base": {}, "lora": {}}

    # Base model eval.
    base = DepthAnythingV2Model(device=config["device"])
    for name, samples in samples_by_ds.items():
        print(f"\n=== base | {name} ({len(samples)} samples) ===")
        results["base"][name] = _evaluate(base, samples, config)
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # LoRA-adapted eval.
    lora_model = DAV2WithLoRA(
        adapter_path=args.adapter_path,
        rank=int(args.rank),
        alpha=int(config["lora_alpha"]),
        dropout=float(config["lora_dropout"]),
        device=config["device"],
    )
    for name, samples in samples_by_ds.items():
        print(f"\n=== lora | {name} ({len(samples)} samples) ===")
        results["lora"][name] = _evaluate(lora_model, samples, config)
    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Forgetting deltas.
    deltas: Dict[str, Dict[str, float]] = {}
    for name in samples_by_ds:
        b = results["base"].get(name, {})
        l = results["lora"].get(name, {})
        deltas[name] = {
            f"d_{m}": float(l.get(m, float("nan")) - b.get(m, float("nan")))
            for m in ("abs_rel", "rmse", "delta1")
        }
    out = {"results": results, "deltas": deltas, "adapter_path": args.adapter_path}
    out_path = os.path.join(output_dir, "forgetting.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nWrote {out_path}")
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
