"""PyTorch dataset / dataloader for viewpoint-bucketed depth data.

Reads samples written by ``dataset_builder.py`` (per-category JSON files) and
returns ``{"image", "depth", "mask", "meta"}`` dicts. Handles RGB images and
the two depth on-disk formats used by TartanAir (.npy, meters) and Hypersim
(.hdf5, meters in the ``dataset`` key).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_image(path: str) -> np.ndarray:
    """Load an RGB image as a uint8 HxWx3 numpy array."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _load_depth(path: str) -> np.ndarray:
    """Load a depth map in meters as a float32 HxW numpy array.

    Supports .npy (TartanAir), .hdf5 (Hypersim), and .png (KITTI 16-bit
    depth-completion format, divided by 256).
    """
    p = str(path)
    if p.endswith(".npy"):
        depth = np.load(p)
    elif p.endswith(".hdf5") or p.endswith(".h5"):
        import h5py
        with h5py.File(p, "r") as f:
            key = "dataset" if "dataset" in f else list(f.keys())[0]
            depth = np.asarray(f[key][:], dtype=np.float32)
    elif p.endswith(".png") or p.endswith(".tif") or p.endswith(".tiff"):
        with Image.open(p) as im:
            arr = np.asarray(im)
        if arr.dtype == np.uint16:
            depth = arr.astype(np.float32) / 256.0  # KITTI convention
        else:
            depth = arr.astype(np.float32)
    elif p.endswith(".pfm"):
        depth = _load_pfm(p)
    else:
        raise ValueError(f"Unsupported depth format: {path}")
    depth = depth.astype(np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth


def _load_pfm(path: str) -> np.ndarray:
    """Read a Middlebury-style PFM file."""
    with open(path, "rb") as f:
        header = f.readline().decode("latin-1").strip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file: {path}")
        color = header == "PF"
        dims = f.readline().decode("latin-1").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("latin-1").strip()
        w, h = (int(x) for x in dims.split())
        scale = float(f.readline().decode("latin-1").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
    shape = (h, w, 3) if color else (h, w)
    data = np.reshape(data, shape)
    return np.flipud(data)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ViewpointDepthDataset(Dataset):
    """Dataset that yields (image, depth, mask, meta) records.

    Parameters
    ----------
    samples: list of dict
        Each dict must contain ``image_path`` and ``depth_path`` keys; other
        keys (``pitch_deg``, ``category``, ``source``, ``env_tag``) are
        forwarded into the ``meta`` field.
    image_size: int
        Output square side length. Inputs are resized with bilinear
        interpolation; depth with nearest-neighbour to avoid bleeding.
    normalize: bool
        Whether to apply ImageNet normalization to the image tensor.
    min_depth, max_depth: float
        Depths outside [min_depth, max_depth] are masked out (mask=0).
    augment: bool
        Whether to apply training-time augmentations (horizontal flip).
    """

    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        image_size: int = 518,
        normalize: bool = True,
        min_depth: float = 0.1,
        max_depth: float = 80.0,
        augment: bool = False,
    ):
        self.samples: List[Dict[str, Any]] = list(samples)
        self.image_size = int(image_size)
        self.normalize = bool(normalize)
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.augment = bool(augment)

        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_split(
        cls,
        split_path: str,
        subset: str = "eval",
        **kwargs: Any,
    ) -> "ViewpointDepthDataset":
        """Build a dataset from a split JSON file produced by DatasetBuilder."""
        with open(split_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if subset not in data:
            raise KeyError(f"subset={subset!r} not in {split_path}")
        return cls(samples=data[subset], **kwargs)

    # ------------------------------------------------------------------
    # Item construction
    # ------------------------------------------------------------------
    def _resize(
        self,
        image: np.ndarray,
        depth: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image (bilinear) and depth (nearest) to ``image_size``."""
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_t = F.interpolate(
            img_t.unsqueeze(0),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        depth_t = torch.from_numpy(depth).float().unsqueeze(0).unsqueeze(0)
        depth_t = F.interpolate(
            depth_t,
            size=(self.image_size, self.image_size),
            mode="nearest",
        ).squeeze(0)
        return img_t, depth_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        try:
            image_np = _load_image(sample["image_path"])
            depth_np = _load_depth(sample["depth_path"])
        except Exception as exc:
            raise RuntimeError(f"Failed to load sample {idx} ({sample}): {exc}")

        # Replace inf / nan / non-positive depths with 0 (will be masked out).
        depth_np = np.where(np.isfinite(depth_np), depth_np, 0.0)
        depth_np = np.clip(depth_np, 0.0, 1e6)

        image_t, depth_t = self._resize(image_np, depth_np)

        if self.augment and torch.rand(1).item() < 0.5:
            image_t = torch.flip(image_t, dims=[2])
            depth_t = torch.flip(depth_t, dims=[2])

        mask = (depth_t > self.min_depth) & (depth_t < self.max_depth)
        mask = mask.float()

        if self.normalize:
            image_t = self._normalize(image_t)

        meta = {
            "image_path": sample.get("image_path", ""),
            "depth_path": sample.get("depth_path", ""),
            "pitch_deg": float(sample.get("pitch_deg", 0.0)),
            "category": str(sample.get("category", "")),
            "source": str(sample.get("source", "")),
            "env_tag": str(sample.get("env_tag", "standard")),
            "index": idx,
        }
        return {
            "image": image_t,
            "depth": depth_t,
            "mask": mask,
            "meta": meta,
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Build a DataLoader with sensible defaults for this project."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def load_category_dataloader(
    split_dir: str,
    category: str,
    subset: str = "eval",
    image_size: int = 518,
    batch_size: int = 8,
    num_workers: int = 4,
    augment: bool = False,
    normalize: bool = True,
    min_depth: float = 0.1,
    max_depth: float = 80.0,
) -> DataLoader:
    """Convenience: build a DataLoader for one viewpoint category split."""
    split_path = os.path.join(split_dir, f"category_{category}.json")
    ds = ViewpointDepthDataset.from_split(
        split_path,
        subset=subset,
        image_size=image_size,
        augment=augment,
        normalize=normalize,
        min_depth=min_depth,
        max_depth=max_depth,
    )
    return make_dataloader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(subset == "finetune"),
    )
