"""Dataset builder for MDE viewpoint benchmarking.

Filters TartanAir and Hypersim frames by camera pitch angle into 5 categories
covering eye-level through top-down views, plus a non-standard environment
category (fog / low light from TartanAir weather variants).

Outputs per-category JSON splits of the form:
    {"eval": [sample, ...], "finetune": [sample, ...]}
where each sample is a dict:
    {"image_path": str, "depth_path": str, "pitch_deg": float,
     "category": str, "source": str, "env_tag": str}
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# Environment tags that mark "non-standard" TartanAir variants for category E.
NONSTANDARD_ENV_TAGS = {
    "fog", "night", "dark", "rain", "snow", "lowlight", "low_light",
    "foggy", "rainy", "snowy",
}


@dataclass
class Sample:
    """A single (image, depth, pose) record."""

    image_path: str
    depth_path: str
    pitch_deg: float
    category: str = ""
    source: str = ""           # "tartanair" or "hypersim"
    env_tag: str = "standard"  # e.g. "standard", "fog", "night"

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pose helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_to_pitch_deg(R: np.ndarray) -> float:
    """Extract pitch from a TartanAir camera-to-world rotation matrix.

    TartanAir poses encode yaw/pitch/roll as ZYX Euler angles applied to
    the camera-to-world transform, so pitch = asin(-R[2,0]).  The camera's
    optical axis is +Z in camera space; pitching the camera down rotates
    around world-Y, which is reflected in R[2,0].
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = math.atan2(-R[2, 0], sy)
    return float(np.clip(abs(math.degrees(pitch)), 0.0, 90.0))


def _hypersim_pitch_deg(R: np.ndarray) -> float:
    """Extract pitch from a Hypersim camera-to-world rotation matrix.

    Hypersim uses a Y-up world frame and stores R_c2w (camera-to-world).
    The camera looks along its local -Z axis, so the look direction in
    world space is ``-R[:, 2]`` (negated third column).  Pitch is the
    angle between that look direction and the horizontal plane:
        pitch = asin(|look_dir · world_up|) = asin(|R[1, 2]|)
    This correctly handles all tilt axes — unlike the TartanAir formula
    which only detects Y-axis (yaw-coupled) tilts and returns ~0 for
    cameras that tilt purely around their own X-axis.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    # |Y-component of camera look direction in world frame|
    sin_pitch = float(np.clip(abs(R[1, 2]), 0.0, 1.0))
    return float(np.clip(abs(math.degrees(math.asin(sin_pitch))), 0.0, 90.0))


def _quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """Convert a (qx, qy, qz, qw) quaternion to a 3x3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    qx, qy, qz, qw = q
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-12:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),         1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),         2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])
    return R


# ---------------------------------------------------------------------------
# DatasetBuilder
# ---------------------------------------------------------------------------

class DatasetBuilder:
    """Discover frames from TartanAir / Hypersim and bucket them by pitch.

    Parameters
    ----------
    config: dict
        Loaded YAML config containing ``pitch_bins``, sample budgets and seed.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.pitch_bins: Dict[str, Tuple[float, float]] = {
            k: (float(v[0]), float(v[1])) for k, v in config["pitch_bins"].items()
        }
        self.n_eval = int(config.get("n_eval_per_category", 1500))
        self.n_ft = int(config.get("n_finetune_per_category", 500))
        self.seed = int(config.get("seed", 42))
        self._frames: List[Sample] = []

    # ------------------------------------------------------------------
    # TartanAir
    # ------------------------------------------------------------------
    def build_tartanair(self, root_dir: str) -> pd.DataFrame:
        """Index a TartanAir directory tree into a DataFrame.

        Expected layout (TartanAir v1):
            <root>/<scene>/<difficulty>/<traj>/image_left/000000_left.png
            <root>/<scene>/<difficulty>/<traj>/depth_left/000000_left_depth.npy
            <root>/<scene>/<difficulty>/<traj>/pose_left.txt
        Pose file: each line is ``tx ty tz qx qy qz qw``.

        Scene folders that contain a ``weather`` keyword (fog / rain / etc.)
        get an env_tag other than "standard" and feed category E.
        """
        root = Path(root_dir)
        rows: List[Sample] = []
        if not root.exists():
            print(f"[TartanAir] root {root} not found — returning empty index.")
            return pd.DataFrame()

        for traj_dir in root.glob("*/*/*"):
            if not traj_dir.is_dir():
                continue
            img_dir = traj_dir / "image_left"
            depth_dir = traj_dir / "depth_left"
            pose_file = traj_dir / "pose_left.txt"
            if not (img_dir.is_dir() and depth_dir.is_dir() and pose_file.is_file()):
                continue

            try:
                poses = np.loadtxt(pose_file)
            except Exception as exc:
                print(f"[TartanAir] failed to read {pose_file}: {exc}")
                continue
            if poses.ndim == 1:
                poses = poses.reshape(1, -1)

            scene_name = traj_dir.parts[-3].lower()
            env_tag = "standard"
            for tag in NONSTANDARD_ENV_TAGS:
                if tag in scene_name:
                    env_tag = tag
                    break

            images = sorted(img_dir.glob("*_left.png"))
            for idx, img_path in enumerate(images):
                if idx >= len(poses):
                    break
                depth_path = depth_dir / f"{img_path.stem}_depth.npy"
                if not depth_path.exists():
                    continue
                R = _quaternion_to_rotation(poses[idx, 3:7])
                pitch = _rotation_matrix_to_pitch_deg(R)
                rows.append(
                    Sample(
                        image_path=str(img_path.resolve()),
                        depth_path=str(depth_path.resolve()),
                        pitch_deg=pitch,
                        source="tartanair",
                        env_tag=env_tag,
                    )
                )

        df = pd.DataFrame([s.to_dict() for s in rows])
        self._frames.extend(rows)
        print(f"[TartanAir] indexed {len(df)} frames from {root}")
        return df

    # ------------------------------------------------------------------
    # Hypersim
    # ------------------------------------------------------------------
    def build_hypersim(self, root_dir: str) -> pd.DataFrame:
        """Index a Hypersim directory tree into a DataFrame.

        Expected layout (Hypersim v1, after dataset preview download):
            <root>/<scene>/images/scene_cam_<k>_final_preview/frame.<i>.color.jpg
            <root>/<scene>/images/scene_cam_<k>_geometry_hdf5/frame.<i>.depth_meters.hdf5
            <root>/<scene>/_detail/<cam>/camera_keyframe_orientations.hdf5
        """
        try:
            import h5py  # imported lazily because hypersim is optional
        except ImportError:
            print("[Hypersim] h5py not installed — skipping Hypersim indexing.")
            return pd.DataFrame()

        root = Path(root_dir)
        rows: List[Sample] = []
        if not root.exists():
            print(f"[Hypersim] root {root} not found — returning empty index.")
            return pd.DataFrame()

        for scene_dir in root.iterdir():
            if not scene_dir.is_dir():
                continue
            detail_dir = scene_dir / "_detail"
            images_dir = scene_dir / "images"
            if not (detail_dir.is_dir() and images_dir.is_dir()):
                continue

            for cam_dir in detail_dir.iterdir():
                if not cam_dir.is_dir():
                    continue
                cam_name = cam_dir.name  # e.g. cam_00
                ori_file = cam_dir / "camera_keyframe_orientations.hdf5"
                if not ori_file.exists():
                    continue
                try:
                    with h5py.File(ori_file, "r") as f:
                        orientations = f["dataset"][:]  # (N, 3, 3)
                except Exception as exc:
                    print(f"[Hypersim] cannot read {ori_file}: {exc}")
                    continue

                color_dir = images_dir / f"scene_{cam_name}_final_preview"
                depth_dir = images_dir / f"scene_{cam_name}_geometry_hdf5"
                if not (color_dir.is_dir() and depth_dir.is_dir()):
                    continue

                color_files = sorted(color_dir.glob("frame.*.color.jpg"))
                for img_path in color_files:
                    try:
                        idx = int(img_path.stem.split(".")[1])
                    except (IndexError, ValueError):
                        continue
                    if idx >= len(orientations):
                        continue
                    depth_path = depth_dir / f"frame.{idx:04d}.depth_meters.hdf5"
                    if not depth_path.exists():
                        continue
                    R = orientations[idx]
                    pitch = _hypersim_pitch_deg(R)
                    rows.append(
                        Sample(
                            image_path=str(img_path.resolve()),
                            depth_path=str(depth_path.resolve()),
                            pitch_deg=pitch,
                            source="hypersim",
                            env_tag="standard",
                        )
                    )

        df = pd.DataFrame([s.to_dict() for s in rows])
        self._frames.extend(rows)
        print(f"[Hypersim] indexed {len(df)} frames from {root}")
        return df

    # ------------------------------------------------------------------
    # Filtering & splitting
    # ------------------------------------------------------------------
    def filter_by_pitch(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Return rows whose pitch lies in the configured bin for ``category``.

        Category E is special: it ignores pitch and instead selects rows whose
        ``env_tag`` indicates a non-standard environment.
        """
        if df.empty:
            return df.copy()

        if category == "E":
            mask = df["env_tag"].isin(NONSTANDARD_ENV_TAGS)
            sub = df[mask].copy()
        else:
            if category not in self.pitch_bins:
                raise KeyError(f"Unknown category {category!r}")
            lo, hi = self.pitch_bins[category]
            mask = (df["pitch_deg"] >= lo) & (df["pitch_deg"] <= hi)
            sub = df[mask].copy()
        sub["category"] = category
        return sub

    def _sample_split(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Shuffle ``df`` deterministically and slice into eval / finetune.

        With enough data, allocates ``n_eval`` to eval and ``n_ft`` to finetune.
        When the bucket is too small to hit both budgets, falls back to a 75/25
        proportional split so finetune is never starved of samples.
        """
        if df.empty:
            return [], []
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        total = len(df)
        budget = self.n_eval + self.n_ft
        if total >= budget:
            n_eval = self.n_eval
        else:
            # Proportional fallback: 75% eval, 25% finetune (with a min of 1 each).
            n_eval = max(1, min(total - 1, int(round(0.75 * total))))
        eval_part = df.iloc[:n_eval].to_dict(orient="records")
        ft_end = min(n_eval + self.n_ft, total)
        ft_part = df.iloc[n_eval:ft_end].to_dict(orient="records")
        return eval_part, ft_part

    def save_splits(self, output_dir: str) -> Dict[str, Dict[str, int]]:
        """Materialize and write per-category JSON split files.

        Returns a small summary dict with sizes per category.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([s.to_dict() for s in self._frames])
        if df.empty:
            print("[save_splits] no frames indexed — writing empty splits.")
        summary: Dict[str, Dict[str, int]] = {}
        for cat in list(self.pitch_bins.keys()) + ["E"]:
            sub = self.filter_by_pitch(df, cat)
            eval_list, ft_list = self._sample_split(sub)
            split = {"eval": eval_list, "finetune": ft_list}
            with open(out_dir / f"category_{cat}.json", "w", encoding="utf-8") as fh:
                json.dump(split, fh, indent=2)
            summary[cat] = {"eval": len(eval_list), "finetune": len(ft_list)}
            print(f"[save_splits] {cat}: eval={len(eval_list)} finetune={len(ft_list)}")
        with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build viewpoint-bucketed splits.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tartanair_root", default=None)
    parser.add_argument("--hypersim_root", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    _seed_everything(int(config.get("seed", 42)))

    tartanair_root = args.tartanair_root or config["tartanair_root"]
    hypersim_root = args.hypersim_root or config["hypersim_root"]
    out_dir = args.output_dir or config["split_dir"]

    builder = DatasetBuilder(config)
    builder.build_tartanair(tartanair_root)
    builder.build_hypersim(hypersim_root)
    builder.save_splits(out_dir)


if __name__ == "__main__":
    main()
