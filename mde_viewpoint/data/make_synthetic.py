"""Generate a tiny TartanAir-shaped synthetic dataset for smoke-testing.

Produces frames at multiple pitch angles (covering categories A-E) plus a
"fog" weather variant for category E. Each frame is a procedurally generated
RGB image + a depth map in meters, with a pose file containing the camera
quaternion. The dataset_builder will discover them via its standard
TartanAir layout walker.

Output layout:
    <root>/<scene>/<difficulty>/<traj>/image_left/000000_left.png
    <root>/<scene>/<difficulty>/<traj>/depth_left/000000_left_depth.npy
    <root>/<scene>/<difficulty>/<traj>/pose_left.txt

Usage:
    python -m mde_viewpoint.data.make_synthetic \
        --root ./datasets/tartanair --frames_per_traj 30
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


def _pitch_quat(pitch_deg: float) -> np.ndarray:
    """Quaternion (qx, qy, qz, qw) for a pure pitch rotation about world Y-axis.

    This matches the pitch convention used by ``dataset_builder._rotation_matrix_to_pitch_deg``
    (which extracts pitch as ``asin(-R[2, 0])``), so the builder will recover the
    intended angle when it walks these synthetic poses.
    """
    p = math.radians(pitch_deg) / 2.0
    return np.array([0.0, math.sin(p), 0.0, math.cos(p)], dtype=np.float64)


def _make_rgb(h: int, w: int, seed: int, fog: bool = False) -> np.ndarray:
    """Build a procedural RGB image with smooth color gradients + texture."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")
    base_r = (0.5 + 0.5 * np.sin(6 * np.pi * xx + rng.uniform()))
    base_g = (0.5 + 0.5 * np.sin(6 * np.pi * yy + rng.uniform()))
    base_b = (0.5 + 0.5 * np.cos(6 * np.pi * (xx + yy) + rng.uniform()))
    img = np.stack([base_r, base_g, base_b], axis=-1)
    img += 0.05 * rng.standard_normal(img.shape)
    img = np.clip(img, 0, 1)
    if fog:
        # Wash colors out + reduce contrast for "fog" variant.
        img = 0.5 * img + 0.5 * np.array([0.78, 0.82, 0.86])
        img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def _make_depth(h: int, w: int, pitch_deg: float, seed: int) -> np.ndarray:
    """Build a plausible depth map (meters) for a given camera pitch.

    - Eye-level (low pitch): vertical gradient (closer at bottom).
    - Top-down (high pitch): nearly uniform depth ~ camera height.
    - Intermediate pitch: blended.
    """
    rng = np.random.default_rng(seed + 1000)
    yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing="ij")

    # Eye-level: ground plane viewed obliquely → 1/(distance from horizon).
    eye_level = 1.0 + 30.0 * (1.0 - yy)  # 1m at bottom, 31m at top

    # Top-down: ~uniform 5m, slight perlin-ish bumps.
    bumps = 0.3 * (np.sin(8 * np.pi * xx) * np.sin(8 * np.pi * yy))
    top_down = 5.0 + bumps

    t = np.clip(pitch_deg / 90.0, 0.0, 1.0)
    depth = (1 - t) * eye_level + t * top_down
    depth += 0.05 * rng.standard_normal(depth.shape)
    return np.clip(depth, 0.5, 80.0).astype(np.float32)


def _write_traj(
    traj_dir: Path,
    n_frames: int,
    pitch_deg: float,
    image_size: int,
    seed: int,
    fog: bool = False,
) -> None:
    """Write one trajectory worth of (image, depth, pose) files."""
    img_dir = traj_dir / "image_left"
    dep_dir = traj_dir / "depth_left"
    img_dir.mkdir(parents=True, exist_ok=True)
    dep_dir.mkdir(parents=True, exist_ok=True)

    quat = _pitch_quat(pitch_deg)
    poses = []
    for i in range(n_frames):
        rgb = _make_rgb(image_size, image_size, seed=seed + i, fog=fog)
        depth = _make_depth(image_size, image_size, pitch_deg=pitch_deg, seed=seed + i)
        stem = f"{i:06d}_left"
        Image.fromarray(rgb).save(img_dir / f"{stem}.png")
        np.save(dep_dir / f"{stem}_depth.npy", depth)
        # tx ty tz qx qy qz qw — translation is irrelevant for our pitch test.
        poses.append([0.0, 0.0, 1.5 + 0.05 * i,
                      quat[0], quat[1], quat[2], quat[3]])
    np.savetxt(traj_dir / "pose_left.txt", np.array(poses), fmt="%.6f")


def make_dataset(
    root: str,
    frames_per_traj: int = 30,
    image_size: int = 256,
    seed: int = 42,
) -> None:
    """Populate ``root`` with synthetic TartanAir-shaped trajectories.

    One scene per category (A,B,C,D + a 'fog_E' scene), one difficulty
    folder per scene, one trajectory per difficulty.
    """
    root_path = Path(root)
    pitch_for_scene = {
        "scene_A_eye_level": 7.5,
        "scene_B_oblique":   37.5,
        "scene_C_steep":     60.0,
        "scene_D_topdown":   85.0,
        "scene_fog_E":       7.5,
    }
    fog_scenes = {"scene_fog_E"}

    rng_seed = seed
    for scene, pitch in pitch_for_scene.items():
        traj_dir = root_path / scene / "Easy" / "P000"
        _write_traj(
            traj_dir=traj_dir,
            n_frames=frames_per_traj,
            pitch_deg=pitch,
            image_size=image_size,
            seed=rng_seed,
            fog=(scene in fog_scenes),
        )
        rng_seed += frames_per_traj
        print(f"[synthetic] wrote {scene}: pitch={pitch}° frames={frames_per_traj}")

    print(f"[synthetic] dataset ready under {root_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TartanAir-shaped data.")
    parser.add_argument("--root", default="./datasets/tartanair",
                        help="Root directory to write into.")
    parser.add_argument("--frames_per_traj", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make_dataset(
        root=args.root,
        frames_per_traj=args.frames_per_traj,
        image_size=args.image_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
