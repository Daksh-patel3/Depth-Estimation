"""Unified wrappers around 4 monocular depth estimation foundation models.

Every wrapper exposes a single ``predict(image)`` method that takes a 4D
``[B, 3, H, W]`` tensor (ImageNet-normalized RGB by convention) and returns
a 4D ``[B, 1, H, W]`` depth tensor at the input spatial resolution. Wrappers
denormalize / re-normalize internally as needed for each backbone.

Models
------
- ``DepthAnythingV2Model``  : HF ``depth-anything/Depth-Anything-V2-Large-hf``
                              (relative depth → median-scaled at eval time).
- ``ZoeDepthModel``         : torch.hub ``isl-org/ZoeDepth`` ZoeD_NK (metric).
- ``MiDaSModel``            : torch.hub ``intel-isl/MiDaS`` DPT_Large (relative).
- ``MarigoldModel``         : diffusers ``prs-eth/marigold-depth-lcm-v1-0``
                              (affine-invariant relative depth).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _denormalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization, returning a tensor in [0, 1]."""
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def _to_pil_batch(x: torch.Tensor) -> List[Image.Image]:
    """Convert a [B, 3, H, W] [0, 1] tensor to a list of PIL images."""
    x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    return [Image.fromarray(x[i].transpose(1, 2, 0)) for i in range(x.shape[0])]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class DepthModel(ABC):
    """Abstract base class for depth-estimation model wrappers.

    Subclasses must implement ``_load_model`` and ``predict``. The class also
    advertises whether its outputs are metric or relative depth via the
    ``is_metric`` attribute, which the evaluator uses to decide whether to
    apply median scaling.
    """

    name: str = "base"
    is_metric: bool = False

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.nn.Module] = None
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Instantiate ``self.model`` and move it to ``self.device``."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Run inference. ``image`` is [B, 3, H, W] ImageNet-normalized."""
        raise NotImplementedError

    def to(self, device: str) -> "DepthModel":
        """Move the underlying model to ``device``."""
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self

    def eval(self) -> "DepthModel":
        if self.model is not None:
            self.model.eval()
        return self


# ---------------------------------------------------------------------------
# Depth Anything V2
# ---------------------------------------------------------------------------

class DepthAnythingV2Model(DepthModel):
    """Wrapper around the HuggingFace Depth Anything V2 large checkpoint."""

    name = "dav2"
    is_metric = False
    hf_id = "depth-anything/Depth-Anything-V2-Large-hf"

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.hf_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict relative depth maps at the input resolution."""
        b, _, h, w = image.shape
        rgb = _denormalize_imagenet(image)
        # The HF processor expects PIL inputs; for batched usage we feed numpy.
        inputs = self.processor(
            images=_to_pil_batch(rgb), return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        pred = out.predicted_depth  # [B, h', w'], DPT-style relative depth
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        # Depth Anything's output is "inverse depth" style — larger=closer. Flip
        # to a depth-like quantity so larger=farther, then median-scale at eval.
        pred = pred.max() - pred + 1e-6
        return pred


# ---------------------------------------------------------------------------
# ZoeDepth
# ---------------------------------------------------------------------------

class ZoeDepthModel(DepthModel):
    """Wrapper around ZoeDepth (NYU+KITTI head, metric depth).

    Uses the HuggingFace checkpoint ``Intel/zoedepth-nyu-kitti``, which is
    actively maintained and avoids the torch.hub state_dict drift seen with
    the original isl-org/ZoeDepth release.
    """

    name = "zoedepth"
    is_metric = True
    hf_id = "Intel/zoedepth-nyu-kitti"

    def _load_model(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(self.hf_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.hf_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        rgb = _denormalize_imagenet(image)
        inputs = self.processor(
            images=_to_pil_batch(rgb), return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        pred = out.predicted_depth
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred


# ---------------------------------------------------------------------------
# MiDaS
# ---------------------------------------------------------------------------

class MiDaSModel(DepthModel):
    """Wrapper around MiDaS DPT_Large (relative depth)."""

    name = "midas"
    is_metric = False

    def _load_model(self) -> None:
        self.model = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = transforms.dpt_transform
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        rgb = _denormalize_imagenet(image)  # in [0, 1]
        rgb_np = (rgb.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        batch_in = []
        for i in range(b):
            tens = self.transform(rgb_np[i])  # returns [1, 3, H, W] tensor
            if isinstance(tens, dict):
                tens = tens["image"]
            if tens.ndim == 3:
                tens = tens.unsqueeze(0)
            batch_in.append(tens)
        batch_in_t = torch.cat(batch_in, dim=0).to(self.device)
        out = self.model(batch_in_t)
        if out.ndim == 3:
            out = out.unsqueeze(1)
        # MiDaS outputs inverse depth — convert to depth-like (larger=farther).
        out = 1.0 / out.clamp(min=1e-6)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Marigold
# ---------------------------------------------------------------------------

class MarigoldModel(DepthModel):
    """Wrapper around the Marigold LCM depth diffusion pipeline."""

    name = "marigold"
    is_metric = False
    hf_id = "prs-eth/marigold-depth-lcm-v1-0"

    def _load_model(self) -> None:
        from diffusers import DiffusionPipeline

        self.pipeline = DiffusionPipeline.from_pretrained(
            self.hf_id,
            custom_pipeline="marigold_depth_estimation",
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        try:
            self.pipeline.to(self.device)
        except Exception:
            pass
        self.model = self.pipeline  # for the .to / .eval API contract

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        b, _, h, w = image.shape
        rgb = _denormalize_imagenet(image)
        pil_imgs = _to_pil_batch(rgb)
        outputs = []
        for pil in pil_imgs:
            res = self.pipeline(
                pil,
                denoising_steps=4,
                ensemble_size=1,
                processing_res=min(self.image_size if hasattr(self, "image_size") else 768, 768),
                match_input_res=True,
                show_progress_bar=False,
            )
            d = res.depth_np if hasattr(res, "depth_np") else np.asarray(res["depth_np"])
            outputs.append(torch.from_numpy(d).float())
        pred = torch.stack(outputs, dim=0).unsqueeze(1)  # [B, 1, H, W]
        pred = pred.to(self.device)
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        return pred


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "dav2": DepthAnythingV2Model,
    "zoedepth": ZoeDepthModel,
    "midas": MiDaSModel,
    "marigold": MarigoldModel,
}


def build_model(name: str, device: str = "cuda") -> DepthModel:
    """Instantiate one of the supported depth models by short name."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; pick from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](device=device)
