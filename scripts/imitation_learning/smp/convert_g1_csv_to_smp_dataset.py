#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert 36-column G1 CSV motions into 192D SMP frames datasets.

The CSV layout is intentionally matched to the LaFAN preprocessing script:
``root_pos(3) + root_quat_xyzw(4) + joint_pos(29)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

try:
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp
except ImportError:  # pragma: no cover - runtime fallback for Isaac Sim python
    CubicSpline = None
    Rotation = None
    Slerp = None


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PATH_BOOTSTRAP = _load_module(
    "isaaclab_smp_csv_path_bootstrap",
    Path(__file__).resolve().with_name("path_bootstrap.py"),
)
_PATH_BOOTSTRAP.prepend_local_source_paths(
    __file__, package_names=("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")
)
_REPO_ROOT = _PATH_BOOTSTRAP.find_repo_root(
    __file__, package_names=("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_tasks")
)

_LUCAS_RSL_RL_ROOT = Path(os.environ.get("LUCAS_RSL_RL_ROOT", "/home/hjqsyy/lucas_rsl_rl"))
if (_LUCAS_RSL_RL_ROOT / "rsl_rl" / "__init__.py").exists() and str(_LUCAS_RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_LUCAS_RSL_RL_ROOT))

_EXPORT_DATASET_MODULE = _load_module(
    "isaaclab_smp_csv_export_dataset",
    Path(__file__).resolve().with_name("export_g1_motion_dataset.py"),
)

build_smp_feature_components = _EXPORT_DATASET_MODULE.build_smp_feature_components
pack_smp_frame_features = _EXPORT_DATASET_MODULE.pack_smp_frame_features
g1_ee_names = list(_EXPORT_DATASET_MODULE.g1_ee_names)
g1_smp_feature_dim = int(_EXPORT_DATASET_MODULE.g1_smp_feature_dim)
g1_smp_joint_axes = [tuple(axis) for axis in _EXPORT_DATASET_MODULE.g1_smp_joint_axes]
g1_smp_joint_names = list(_EXPORT_DATASET_MODULE.g1_smp_joint_names)
g1_smp_window_size = int(_EXPORT_DATASET_MODULE.g1_smp_window_size)
_build_default_joint_pos = _EXPORT_DATASET_MODULE._build_default_joint_pos

_CSV_NUM_COLUMNS = 36
_CSV_ROOT_POS_SLICE = slice(0, 3)
_CSV_ROOT_QUAT_XYZW_SLICE = slice(3, 7)
_CSV_JOINT_POS_SLICE = slice(7, 36)
_DEFAULT_G1_USD_CANDIDATES = (
    Path("/home/hjqsyy/IsaacLab/unitree_rl_lab/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"),
    Path("/home/hjqsyy/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/usd/g1_29dof_rev_1_0.usd"),
    Path("/home/hjqsyy/.cache/huggingface/hub/datasets--unitreerobotics--unitree_model/snapshots/c1a18103c7f551bff0fc5a2ba8a739adc067d44e/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd"),
)
_DEFAULT_G1_URDF_CANDIDATES = (
    Path("/home/hjqsyy/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/usd/g1_29dof_rev_1_0.urdf"),
    Path("/home/hjqsyy/lafan/usd/g1_29dof_rev_1_0.urdf"),
)


@dataclass(frozen=True)
class G1CsvMotion:
    """Parsed 36-column G1 CSV motion."""

    root_pos: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_pos: np.ndarray
    fps: float

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])


@dataclass(frozen=True)
class ConversionSummary:
    input_path: str
    output_path: str
    raw_frames: int
    exported_frames: int
    fps: float
    feature_dim: int
    clipped_values: int
    verify_windows: int | None


@dataclass(frozen=True)
class UrdfJointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None


def _resolve_g1_usd_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"G1 USD path does not exist: {path}")
        return path

    env_path = os.environ.get("UNITREE_G1_29DOF_USD")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"UNITREE_G1_29DOF_USD does not exist: {path}")

    for path in _DEFAULT_G1_USD_CANDIDATES:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find a G1 29DOF USD. Pass --usd-path or set UNITREE_G1_29DOF_USD."
    )


def _resolve_g1_urdf_path(explicit_path: str | Path | None = None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"G1 URDF path does not exist: {path}")
        return path

    env_path = os.environ.get("UNITREE_G1_29DOF_URDF")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"UNITREE_G1_29DOF_URDF does not exist: {path}")

    for path in _DEFAULT_G1_URDF_CANDIDATES:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "Could not find a G1 29DOF URDF. Pass --urdf-path or set UNITREE_G1_29DOF_URDF."
    )


def load_g1_csv_motion(csv_path: str | Path, fps: float) -> G1CsvMotion:
    """Load a 36-column G1 CSV as root position, root quaternion, and joint positions.

    Column convention is copied from the LaFAN converter:
    - columns 0:3  -> root position
    - columns 3:7  -> root quaternion in xyzw order
    - columns 7:36 -> 29 G1 joint positions in ``g1_smp_joint_names`` order
    """
    csv_path = Path(csv_path)
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[None, :]
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D CSV data from {csv_path}, got shape {raw.shape}")
    if raw.shape[1] != _CSV_NUM_COLUMNS:
        raise ValueError(f"{csv_path.name} must have 36 columns, got {raw.shape[1]}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    return G1CsvMotion(
        root_pos=raw[:, _CSV_ROOT_POS_SLICE].astype(np.float32),
        root_quat_xyzw=raw[:, _CSV_ROOT_QUAT_XYZW_SLICE].astype(np.float32),
        joint_pos=raw[:, _CSV_JOINT_POS_SLICE].astype(np.float32),
        fps=float(fps),
    )


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Root quaternion contains a near-zero norm entry")
    return (quat_xyzw / norms).astype(np.float32)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1).astype(np.float32)


def _quat_conjugate_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    result = quat_xyzw.copy()
    result[..., :3] *= -1.0
    return result


def _quat_multiply_xyzw(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lx, ly, lz, lw = np.moveaxis(lhs, -1, 0)
    rx, ry, rz, rw = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        axis=-1,
    )


def _quat_to_rotvec_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = _normalize_quat_xyzw(quat_xyzw).astype(np.float64)
    if quat_xyzw[..., 3] < 0.0:
        quat_xyzw = -quat_xyzw
    xyz = quat_xyzw[..., :3]
    w = float(np.clip(quat_xyzw[..., 3], -1.0, 1.0))
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1.0e-8:
        return (2.0 * xyz).astype(np.float32)
    axis = xyz / sin_half
    angle = 2.0 * math.atan2(sin_half, w)
    if angle > math.pi:
        angle -= 2.0 * math.pi
    return (axis * angle).astype(np.float32)


def _slerp_pair_xyzw(quat0_xyzw: np.ndarray, quat1_xyzw: np.ndarray, fraction: float) -> np.ndarray:
    quat0 = _normalize_quat_xyzw(quat0_xyzw).astype(np.float64)
    quat1 = _normalize_quat_xyzw(quat1_xyzw).astype(np.float64)
    dot = float(np.dot(quat0, quat1))
    if dot < 0.0:
        quat1 = -quat1
        dot = -dot
    if dot > 0.9995:
        blended = quat0 + fraction * (quat1 - quat0)
        return _normalize_quat_xyzw(blended).astype(np.float32)
    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * fraction
    sin_theta = math.sin(theta)
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * quat0 + s1 * quat1).astype(np.float32)


def _resample_linear_series(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    dims = [np.interp(target_times, source_times, values[:, dim]) for dim in range(values.shape[1])]
    return np.stack(dims, axis=-1).astype(np.float32)


def _resample_quat_series_xyzw(quat_xyzw: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    normalized = _normalize_quat_xyzw(quat_xyzw).astype(np.float64)
    if len(normalized) == 1:
        return np.repeat(normalized.astype(np.float32), len(target_times), axis=0)

    resampled = np.zeros((len(target_times), 4), dtype=np.float32)
    max_index = len(source_times) - 1
    for idx, target_time in enumerate(target_times):
        if target_time <= source_times[0]:
            resampled[idx] = normalized[0].astype(np.float32)
            continue
        if target_time >= source_times[-1]:
            resampled[idx] = normalized[-1].astype(np.float32)
            continue
        right = int(np.searchsorted(source_times, target_time, side='right'))
        left = max(0, right - 1)
        right = min(right, max_index)
        span = source_times[right] - source_times[left]
        fraction = 0.0 if span <= 1.0e-12 else float((target_time - source_times[left]) / span)
        resampled[idx] = _slerp_pair_xyzw(normalized[left], normalized[right], fraction)
    return resampled


def _parse_xyz_attr(text: str | None) -> np.ndarray:
    if not text:
        return np.zeros(3, dtype=np.float64)
    return np.fromstring(text, sep=" ", dtype=np.float64)


def _parse_axis_attr(text: str | None) -> np.ndarray:
    axis = _parse_xyz_attr(text)
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return axis / norm


def _rpy_to_rotmat_xyz(rpy: np.ndarray) -> np.ndarray:
    if Rotation is not None:
        return Rotation.from_euler("xyz", rpy).as_matrix().astype(np.float64)

    roll, pitch, yaw = rpy.astype(np.float64)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _quat_xyzw_to_rotmat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = _normalize_quat_xyzw(quat_xyzw).astype(np.float64)
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat_xyzw.shape[:-1] + (3, 3))


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _batch_axis_angle_to_matrix(angles: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.clip(np.linalg.norm(axis), 1.0e-12, None)
    x, y, z = axis
    cos = np.cos(angles)[:, None, None]
    sin = np.sin(angles)[:, None, None]
    one_minus_cos = 1.0 - cos
    outer = np.array(
        [
            [x * x, x * y, x * z],
            [y * x, y * y, y * z],
            [z * x, z * y, z * z],
        ],
        dtype=np.float64,
    )[None, :, :]
    cross = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )[None, :, :]
    identity = np.eye(3, dtype=np.float64)[None, :, :]
    return cos * identity + one_minus_cos * outer + sin * cross


def resample_g1_csv_motion(motion: G1CsvMotion, target_fps: float | None) -> G1CsvMotion:
    """Resample root position, root quaternion, and joint positions to ``target_fps``."""
    if target_fps is None or math.isclose(float(target_fps), motion.fps):
        return motion
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if motion.num_frames <= 1:
        return G1CsvMotion(
            root_pos=motion.root_pos.copy(),
            root_quat_xyzw=_normalize_quat_xyzw(motion.root_quat_xyzw),
            joint_pos=motion.joint_pos.copy(),
            fps=float(target_fps),
        )

    source_times = np.arange(motion.num_frames, dtype=np.float64) / motion.fps
    duration = float(source_times[-1])
    target_length = int(np.floor(duration * float(target_fps))) + 1
    target_times = np.arange(target_length, dtype=np.float64) / float(target_fps)

    if CubicSpline is not None:
        root_pos = CubicSpline(source_times, motion.root_pos.astype(np.float64), axis=0)(target_times).astype(np.float32)
        joint_pos = CubicSpline(source_times, motion.joint_pos.astype(np.float64), axis=0)(target_times).astype(np.float32)
    else:
        root_pos = _resample_linear_series(motion.root_pos, source_times, target_times)
        joint_pos = _resample_linear_series(motion.joint_pos, source_times, target_times)

    if Rotation is not None and Slerp is not None:
        root_quat_xyzw = Slerp(source_times, Rotation.from_quat(_normalize_quat_xyzw(motion.root_quat_xyzw)))(
            target_times
        ).as_quat().astype(np.float32)
    else:
        root_quat_xyzw = _resample_quat_series_xyzw(motion.root_quat_xyzw, source_times, target_times)

    return G1CsvMotion(root_pos=root_pos, root_quat_xyzw=root_quat_xyzw, joint_pos=joint_pos, fps=float(target_fps))


def compute_root_linear_velocity(root_pos: np.ndarray, fps: float) -> np.ndarray:
    root_pos = np.asarray(root_pos, dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
        raise ValueError(f"Expected root_pos shape (T, 3), got {root_pos.shape}")
    if root_pos.shape[0] <= 1:
        return np.zeros_like(root_pos, dtype=np.float32)
    return np.gradient(root_pos, 1.0 / float(fps), axis=0, edge_order=1).astype(np.float32)


def compute_root_angular_velocity(root_quat_xyzw: np.ndarray, fps: float) -> np.ndarray:
    root_quat_xyzw = _normalize_quat_xyzw(root_quat_xyzw)
    num_frames = root_quat_xyzw.shape[0]
    if num_frames <= 1:
        return np.zeros((num_frames, 3), dtype=np.float32)

    dt = 1.0 / float(fps)
    ang_vel = np.zeros((num_frames, 3), dtype=np.float64)

    if Rotation is not None:
        rotations = Rotation.from_quat(root_quat_xyzw)
        one_step = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / dt
        ang_vel[0] = one_step[0]
        ang_vel[-1] = one_step[-1]
        if num_frames > 2:
            two_step = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2.0 * dt)
            ang_vel[1:-1] = two_step
        return ang_vel.astype(np.float32)

    one_step = _quat_multiply_xyzw(root_quat_xyzw[1:], _quat_conjugate_xyzw(root_quat_xyzw[:-1]))
    ang_vel[0] = _quat_to_rotvec_xyzw(one_step[0]) / dt
    ang_vel[-1] = _quat_to_rotvec_xyzw(one_step[-1]) / dt
    if num_frames > 2:
        two_step = _quat_multiply_xyzw(root_quat_xyzw[2:], _quat_conjugate_xyzw(root_quat_xyzw[:-2]))
        ang_vel[1:-1] = np.stack([_quat_to_rotvec_xyzw(quat) for quat in two_step], axis=0) / (2.0 * dt)
    return ang_vel.astype(np.float32)


def build_smp_frames_from_motion_arrays(
    *,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    ee_pos_w: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Build 192D SMP frame features from pose arrays and FK end-effector positions."""
    root_pos = np.asarray(root_pos, dtype=np.float32)
    root_quat_wxyz = _xyzw_to_wxyz(root_quat_xyzw)
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    ee_pos_w = np.asarray(ee_pos_w, dtype=np.float32)

    if root_pos.ndim != 2 or root_pos.shape[-1] != 3:
        raise ValueError(f"Expected root_pos shape (T, 3), got {root_pos.shape}")
    if joint_pos.ndim != 2 or joint_pos.shape[-1] != len(g1_smp_joint_names):
        raise ValueError(f"Expected joint_pos shape (T, {len(g1_smp_joint_names)}), got {joint_pos.shape}")
    if ee_pos_w.shape != (root_pos.shape[0], len(g1_ee_names), 3):
        raise ValueError(f"Expected ee_pos_w shape (T, {len(g1_ee_names)}, 3), got {ee_pos_w.shape}")

    root_lin_vel_w = compute_root_linear_velocity(root_pos, fps)
    root_ang_vel_w = compute_root_angular_velocity(root_quat_xyzw, fps)

    default_joint_pos = _build_default_joint_pos(g1_smp_joint_names)
    joint_axes = torch.tensor(g1_smp_joint_axes, dtype=torch.float32)
    feature_components = build_smp_feature_components(
        root_pos_w=torch.as_tensor(root_pos, dtype=torch.float32),
        root_quat_w=torch.as_tensor(root_quat_wxyz, dtype=torch.float32),
        root_lin_vel_w=torch.as_tensor(root_lin_vel_w, dtype=torch.float32),
        root_ang_vel_w=torch.as_tensor(root_ang_vel_w, dtype=torch.float32),
        joint_pos=torch.as_tensor(joint_pos, dtype=torch.float32),
        default_joint_pos=default_joint_pos,
        joint_axes=joint_axes,
        ee_pos_w=torch.as_tensor(ee_pos_w, dtype=torch.float32),
    )
    frames = pack_smp_frame_features(
        base_lin_vel_b=feature_components["base_lin_vel_b"],
        base_ang_vel_b=feature_components["base_ang_vel_b"],
        joint_rot6d_rel=feature_components["joint_rot6d_rel"],
        ee_pos_b=feature_components["ee_pos_b"],
        expected_feature_dim=g1_smp_feature_dim,
    )
    return frames.cpu().numpy().astype(np.float32)


def save_smp_frames_dataset(
    *,
    output_path: str | Path,
    frames: np.ndarray,
    fps: float,
    window_size: int,
    stride: int,
    source_name: str | None = None,
    style_name: str | None = None,
    style_id: int | None = None,
    extra_payload: dict[str, np.ndarray] | None = None,
) -> Path:
    """Save an SMP frames dataset compatible with the existing prior training/audit loaders."""
    output_path = Path(output_path)
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Expected frames shape (T, F), got {frames.shape}")
    if frames.shape[-1] != g1_smp_feature_dim:
        raise ValueError(f"Expected SMP feature dim {g1_smp_feature_dim}, got {frames.shape[-1]}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "frames": frames,
        "fps": np.array([fps], dtype=np.float32),
        "window_size": np.array([window_size], dtype=np.int64),
        "stride": np.array([stride], dtype=np.int64),
        "feature_dim": np.array([frames.shape[-1]], dtype=np.int64),
        "joint_names": np.asarray(g1_smp_joint_names),
        "joint_axes": np.asarray(g1_smp_joint_axes, dtype=np.float32),
        "ee_names": np.asarray(g1_ee_names),
    }
    if source_name is not None:
        payload["source_name"] = np.asarray([source_name], dtype=np.str_)
    if style_name is not None:
        payload["style_name"] = np.asarray([style_name], dtype=np.str_)
    if style_id is not None:
        payload["style_id"] = np.asarray([style_id], dtype=np.int64)
    if extra_payload:
        payload.update(extra_payload)

    np.savez(output_path, **payload)
    return output_path


class G1UrdfForwardKinematics:
    """Pure-URDF G1 FK helper that avoids Isaac Sim runtime startup."""

    def __init__(self, *, urdf_path: str | Path | None = None):
        self.root_link = "pelvis"
        self.urdf_path = _resolve_g1_urdf_path(urdf_path)
        self.joints: dict[str, UrdfJointSpec] = {}
        self.joints_by_child: dict[str, UrdfJointSpec] = {}
        self.ee_chains: dict[str, list[UrdfJointSpec]] = {}
        self.joint_limits: np.ndarray | None = None
        self._setup()

    def _setup(self) -> None:
        root = ET.parse(self.urdf_path).getroot()
        for joint_elem in root.findall("joint"):
            parent_elem = joint_elem.find("parent")
            child_elem = joint_elem.find("child")
            if parent_elem is None or child_elem is None:
                continue
            origin_elem = joint_elem.find("origin")
            axis_elem = joint_elem.find("axis")
            limit_elem = joint_elem.find("limit")
            spec = UrdfJointSpec(
                name=joint_elem.attrib["name"],
                joint_type=joint_elem.attrib.get("type", "fixed"),
                parent=parent_elem.attrib["link"],
                child=child_elem.attrib["link"],
                origin_xyz=_parse_xyz_attr(origin_elem.get("xyz") if origin_elem is not None else None),
                origin_rpy=_parse_xyz_attr(origin_elem.get("rpy") if origin_elem is not None else None),
                axis=_parse_axis_attr(axis_elem.get("xyz") if axis_elem is not None else None),
                lower=float(limit_elem.attrib["lower"]) if limit_elem is not None and "lower" in limit_elem.attrib else None,
                upper=float(limit_elem.attrib["upper"]) if limit_elem is not None and "upper" in limit_elem.attrib else None,
            )
            self.joints[spec.name] = spec
            self.joints_by_child[spec.child] = spec

        dataset_joint_specs = [self.joints[joint_name] for joint_name in g1_smp_joint_names]
        self.joint_limits = np.array(
            [[spec.lower, spec.upper] for spec in dataset_joint_specs],
            dtype=np.float32,
        )
        if np.isnan(self.joint_limits).any():
            raise ValueError("URDF contains missing joint limits for G1 SMP joints.")

        for ee_name in g1_ee_names:
            self.ee_chains[ee_name] = self._build_chain(self.root_link, ee_name)

    def _build_chain(self, root_link: str, target_link: str) -> list[UrdfJointSpec]:
        chain: list[UrdfJointSpec] = []
        current_link = target_link
        while current_link != root_link:
            joint = self.joints_by_child.get(current_link)
            if joint is None:
                raise ValueError(f"Cannot build URDF chain from {root_link} to {target_link}")
            chain.append(joint)
            current_link = joint.parent
        chain.reverse()
        return chain

    def clip_joint_positions(self, joint_pos: np.ndarray) -> tuple[np.ndarray, int]:
        if self.joint_limits is None:
            raise RuntimeError("URDF FK is not initialized")
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        clipped_mask = (joint_pos < lower[None, :]) | (joint_pos > upper[None, :])
        return np.clip(joint_pos, lower[None, :], upper[None, :]).astype(np.float32), int(clipped_mask.sum())

    def compute_ee_positions(self, *, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        root_pos = np.asarray(root_pos, dtype=np.float64)
        root_rotmat = _quat_xyzw_to_rotmat(root_quat_xyzw).astype(np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        num_frames = root_pos.shape[0]
        ee_pos_w = np.zeros((num_frames, len(g1_ee_names), 3), dtype=np.float32)
        joint_angle_map = {joint_name: joint_pos[:, index] for index, joint_name in enumerate(g1_smp_joint_names)}

        for ee_index, ee_name in enumerate(g1_ee_names):
            transforms = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
            transforms[:, :3, :3] = root_rotmat
            transforms[:, :3, 3] = root_pos
            for joint in self.ee_chains[ee_name]:
                origin_tf = _make_transform(_rpy_to_rotmat_xyz(joint.origin_rpy), joint.origin_xyz)
                transforms = transforms @ origin_tf[None, :, :]
                if joint.joint_type in {"revolute", "continuous"}:
                    joint_tf = np.tile(np.eye(4, dtype=np.float64), (num_frames, 1, 1))
                    joint_tf[:, :3, :3] = _batch_axis_angle_to_matrix(joint_angle_map[joint.name], joint.axis)
                    transforms = transforms @ joint_tf
                elif joint.joint_type != "fixed":
                    raise NotImplementedError(f"Unsupported URDF joint type: {joint.joint_type}")
            ee_pos_w[:, ee_index] = transforms[:, :3, 3].astype(np.float32)

        return ee_pos_w

    def close(self) -> None:
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class G1IsaacLabForwardKinematicsBatcher:
    """Batched G1 FK helper backed by IsaacLab articulation kinematics."""

    def __init__(self, *, batch_size: int, device: str, usd_path: str | Path | None, dt: float):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.dt = float(dt)
        self.usd_path = _resolve_g1_usd_path(usd_path)
        self._sim_context = None
        self.sim = None
        self.robot = None
        self.origins = None
        self.ee_body_ids = None
        self.joint_ids = None
        self.joint_limits = None
        self._setup()

    def _setup(self) -> None:
        import isaacsim.core.utils.prims as prim_utils

        from isaaclab.assets import Articulation
        from isaaclab.sim import build_simulation_context
        from isaaclab_assets import UNITREE_G1_29DOF_CFG

        self._sim_context = build_simulation_context(
            create_new_stage=True,
            gravity_enabled=False,
            device=self.device,
            dt=max(self.dt, 1.0e-4),
            add_ground_plane=False,
            add_lighting=False,
            auto_add_lighting=False,
        )
        self.sim = self._sim_context.__enter__()

        self.origins = torch.zeros((self.batch_size, 3), dtype=torch.float32, device=self.sim.device)
        self.origins[:, 0] = torch.arange(self.batch_size, dtype=torch.float32, device=self.sim.device) * 5.0
        for env_id in range(self.batch_size):
            prim_utils.create_prim(f"/World/Env_{env_id}", "Xform", translation=self.origins[env_id].cpu().tolist())

        robot_cfg = UNITREE_G1_29DOF_CFG.copy()
        robot_cfg.prim_path = "/World/Env_.*/Robot"
        robot_cfg.spawn.usd_path = str(self.usd_path)
        self.robot = Articulation(cfg=robot_cfg)
        self.sim.reset()
        self.robot.update(self.sim.get_physics_dt())

        self.ee_body_ids, ee_names = self.robot.find_bodies(g1_ee_names, preserve_order=True)
        self.joint_ids, joint_names = self.robot.find_joints(g1_smp_joint_names, preserve_order=True)
        if list(ee_names) != list(g1_ee_names):
            raise RuntimeError(f"G1 EE body order mismatch: expected {g1_ee_names}, got {ee_names}")
        if list(joint_names) != list(g1_smp_joint_names):
            raise RuntimeError(f"G1 joint order mismatch: expected {g1_smp_joint_names}, got {joint_names}")

        self.joint_limits = self.robot.data.joint_pos_limits[0, self.joint_ids].detach().cpu().numpy().astype(np.float32)

    def clip_joint_positions(self, joint_pos: np.ndarray) -> tuple[np.ndarray, int]:
        if self.joint_limits is None:
            raise RuntimeError("FK batcher is not initialized")
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        clipped_mask = (joint_pos < lower[None, :]) | (joint_pos > upper[None, :])
        return np.clip(joint_pos, lower[None, :], upper[None, :]).astype(np.float32), int(clipped_mask.sum())

    def compute_ee_positions(self, *, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, joint_pos: np.ndarray) -> np.ndarray:
        if self.robot is None or self.sim is None or self.origins is None:
            raise RuntimeError("FK batcher is not initialized")
        root_pos = np.asarray(root_pos, dtype=np.float32)
        root_quat_wxyz = _xyzw_to_wxyz(root_quat_xyzw)
        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        num_frames = root_pos.shape[0]
        ee_pos_w = np.zeros((num_frames, len(g1_ee_names), 3), dtype=np.float32)

        default_root_pose = self.robot.data.default_root_state[:, :7].clone()
        default_joint_pos = self.robot.data.default_joint_pos[:, self.joint_ids].clone()
        zero_root_vel = torch.zeros((self.batch_size, 6), dtype=torch.float32, device=self.sim.device)
        zero_joint_vel = torch.zeros((self.batch_size, len(self.joint_ids)), dtype=torch.float32, device=self.sim.device)

        for start in range(0, num_frames, self.batch_size):
            stop = min(start + self.batch_size, num_frames)
            count = stop - start

            root_pose = default_root_pose.clone()
            root_pose[:, :3] += self.origins
            root_pose[:count, :3] = torch.as_tensor(root_pos[start:stop], dtype=torch.float32, device=self.sim.device)
            root_pose[:count, :3] += self.origins[:count]
            root_pose[:count, 3:7] = torch.as_tensor(root_quat_wxyz[start:stop], dtype=torch.float32, device=self.sim.device)

            joint_state = default_joint_pos.clone()
            joint_state[:count] = torch.as_tensor(joint_pos[start:stop], dtype=torch.float32, device=self.sim.device)

            self.robot.write_root_pose_to_sim(root_pose)
            self.robot.write_root_velocity_to_sim(zero_root_vel)
            self.robot.write_joint_state_to_sim(joint_state, zero_joint_vel, joint_ids=self.joint_ids)
            self.sim.forward()
            self.robot.update(0.0)

            chunk_ee_pos = self.robot.data.body_pos_w[:count, self.ee_body_ids].detach().cpu().numpy()
            chunk_origins = self.origins[:count].detach().cpu().numpy()[:, None, :]
            ee_pos_w[start:stop] = chunk_ee_pos - chunk_origins

        return ee_pos_w

    def close(self) -> None:
        if self._sim_context is not None:
            self._sim_context.__exit__(None, None, None)
            self._sim_context = None
            self.sim = None
            self.robot = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def _verify_smp_dataset(input_path: Path, *, window_size: int, stride: int) -> int:
    diagnostics = _load_module(
        "isaaclab_smp_csv_diagnostics",
        _REPO_ROOT / "source" / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "smp_diagnostics.py",
    )
    windows, _ = diagnostics.load_motion_windows(
        input_path,
        dataset_label="positive",
        window_size=window_size,
        stride=stride,
        converted_dir=None,
    )
    if windows.ndim != 3:
        raise ValueError(f"Expected verification windows ndim 3, got {windows.ndim}")
    if windows.shape[-1] != g1_smp_feature_dim:
        raise ValueError(f"Expected verification feature dim {g1_smp_feature_dim}, got {windows.shape[-1]}")
    return int(windows.shape[0])


def convert_single_csv_to_smp_dataset(
    *,
    csv_path: str | Path,
    output_path: str | Path,
    source_fps: float = 30.0,
    target_fps: float | None = 50.0,
    window_size: int = g1_smp_window_size,
    stride: int = 1,
    fk_batch_size: int = 128,
    device: str = "cpu",
    fk_backend: str = "urdf",
    usd_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    style_name: str | None = None,
    style_id: int | None = None,
    source_name: str | None = None,
    verify: bool = False,
) -> ConversionSummary:
    csv_path = Path(csv_path).resolve()
    output_path = Path(output_path).resolve()
    raw_motion = load_g1_csv_motion(csv_path, fps=source_fps)
    motion = resample_g1_csv_motion(raw_motion, target_fps)

    if fk_backend == "isaaclab":
        fk_ctx = G1IsaacLabForwardKinematicsBatcher(
            batch_size=fk_batch_size,
            device=device,
            usd_path=usd_path,
            dt=1.0 / motion.fps,
        )
    elif fk_backend == "urdf":
        fk_ctx = G1UrdfForwardKinematics(urdf_path=urdf_path)
    else:
        raise ValueError(f"Unsupported fk_backend: {fk_backend}")

    with fk_ctx as fk:
        clipped_joint_pos, clipped_values = fk.clip_joint_positions(motion.joint_pos)
        ee_pos_w = fk.compute_ee_positions(
            root_pos=motion.root_pos,
            root_quat_xyzw=motion.root_quat_xyzw,
            joint_pos=clipped_joint_pos,
        )
        joint_limits = fk.joint_limits.copy()
        resolved_kinematics_path = getattr(fk, "usd_path", None) or getattr(fk, "urdf_path", None)

    frames = build_smp_frames_from_motion_arrays(
        root_pos=motion.root_pos,
        root_quat_xyzw=motion.root_quat_xyzw,
        joint_pos=clipped_joint_pos,
        ee_pos_w=ee_pos_w,
        fps=motion.fps,
    )
    if frames.shape[-1] != g1_smp_feature_dim:
        raise RuntimeError(f"Expected {g1_smp_feature_dim}D frames, got {frames.shape[-1]}")
    if np.isnan(frames).any() or np.isinf(frames).any():
        raise RuntimeError(f"{csv_path.name} produced NaN/Inf frames")

    source_name = source_name or csv_path.stem
    extra_payload = {
        "csv_path": np.asarray([str(csv_path)], dtype=np.str_),
        "source_fps": np.array([source_fps], dtype=np.float32),
        "target_fps": np.array([motion.fps], dtype=np.float32),
        "raw_frames": np.array([raw_motion.num_frames], dtype=np.int64),
        "resampled_frames": np.array([motion.num_frames], dtype=np.int64),
        "clipped_values": np.array([clipped_values], dtype=np.int64),
        "root_pos": motion.root_pos.astype(np.float32),
        "root_quat_xyzw": _normalize_quat_xyzw(motion.root_quat_xyzw),
        "joint_pos": clipped_joint_pos.astype(np.float32),
        "ee_pos_w": ee_pos_w.astype(np.float32),
        "joint_limits": joint_limits.astype(np.float32),
        "fk_backend": np.asarray([fk_backend], dtype=np.str_),
        "kinematics_path": np.asarray([str(resolved_kinematics_path)], dtype=np.str_),
        "csv_column_layout": np.asarray(["root_pos[0:3],root_quat_xyzw[3:7],joint_pos[7:36]"], dtype=np.str_),
    }
    save_smp_frames_dataset(
        output_path=output_path,
        frames=frames,
        fps=motion.fps,
        window_size=window_size,
        stride=stride,
        source_name=source_name,
        style_name=style_name,
        style_id=style_id,
        extra_payload=extra_payload,
    )

    verify_windows = _verify_smp_dataset(output_path, window_size=window_size, stride=stride) if verify else None
    _print_output_summary(output_path, frames, motion.fps, clipped_values, verify_windows)
    return ConversionSummary(
        input_path=str(csv_path),
        output_path=str(output_path),
        raw_frames=raw_motion.num_frames,
        exported_frames=motion.num_frames,
        fps=float(motion.fps),
        feature_dim=int(frames.shape[-1]),
        clipped_values=clipped_values,
        verify_windows=verify_windows,
    )


def convert_csv_directory_to_smp_datasets(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    source_fps: float = 30.0,
    target_fps: float | None = 50.0,
    window_size: int = g1_smp_window_size,
    stride: int = 1,
    fk_batch_size: int = 128,
    device: str = "cpu",
    fk_backend: str = "urdf",
    usd_path: str | Path | None = None,
    urdf_path: str | Path | None = None,
    style_name: str | None = None,
    style_id: int | None = None,
    verify: bool = False,
) -> list[ConversionSummary]:
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for csv_path in csv_files:
        summaries.append(
            convert_single_csv_to_smp_dataset(
                csv_path=csv_path,
                output_path=output_dir / f"{csv_path.stem}.npz",
                source_fps=source_fps,
                target_fps=target_fps,
                window_size=window_size,
                stride=stride,
                fk_batch_size=fk_batch_size,
                device=device,
                fk_backend=fk_backend,
                usd_path=usd_path,
                urdf_path=urdf_path,
                style_name=style_name,
                style_id=style_id,
                source_name=csv_path.stem,
                verify=verify,
            )
        )

    manifest = {
        "datasets": [
            {
                "name": Path(summary.output_path).stem,
                "path": Path(summary.output_path).name,
                "style": style_name,
                "style_id": style_id,
                "weight": 1.0,
            }
            for summary in summaries
        ],
        "style_vocab": {} if style_name is None or style_id is None else {style_name: int(style_id)},
        "window_size": int(window_size),
        "stride": int(stride),
        "feature_dim": int(g1_smp_feature_dim),
    }
    (output_dir / "corpus_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def _print_output_summary(output_path: Path, frames: np.ndarray, fps: float, clipped_values: int, verify_windows: int | None) -> None:
    print(f"output_npz: {output_path}")
    print(f"frames: {frames.shape}")
    print(f"fps: ({fps:g})")
    print(f"feature_dim: {frames.shape[-1]}")
    print(f"clipped_values: {clipped_values}")
    if verify_windows is not None:
        print(f"verify_windows: {verify_windows}")


def _write_report(path: str | Path | None, summaries: list[ConversionSummary]) -> None:
    if path is None:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "num_files": len(summaries),
                "feature_dim": g1_smp_feature_dim,
                "files": [summary.__dict__ for summary in summaries],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert 36-column G1 CSV motions into 192D SMP frames datasets.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-csv", type=Path, help="Single 36-column CSV file.")
    input_group.add_argument("--input-dir", type=Path, help="Directory containing 36-column CSV files.")
    parser.add_argument("--output", type=Path, default=None, help="Output NPZ path for --input-csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for --input-dir.")
    parser.add_argument("--source-fps", type=float, default=30.0, help="CSV source FPS.")
    parser.add_argument("--target-fps", type=float, default=50.0, help="Target FPS after resampling; default matches LaFAN converter.")
    parser.add_argument("--no-resample", action="store_true", help="Keep source FPS instead of resampling to --target-fps.")
    parser.add_argument("--window-size", type=int, default=g1_smp_window_size, help="Window size metadata for audit/training.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window stride metadata for audit/training.")
    parser.add_argument("--fk-backend", choices=("urdf", "isaaclab"), default="urdf", help="FK backend. Default uses pure URDF to avoid Isaac Sim runtime startup.")
    parser.add_argument("--fk-batch-size", type=int, default=128, help="Number of frames evaluated per IsaacLab FK batch.")
    parser.add_argument("--usd-path", type=Path, default=None, help="Override G1 29DOF USD path used for IsaacLab FK.")
    parser.add_argument("--urdf-path", type=Path, default=None, help="Override G1 29DOF URDF path used for pure-URDF FK.")
    parser.add_argument("--style-name", default=None, help="Optional style name metadata.")
    parser.add_argument("--style-id", type=int, default=None, help="Optional style id metadata.")
    parser.add_argument("--source-name", default=None, help="Optional source name for single-file export.")
    parser.add_argument("--verify", action="store_true", help="Verify that smp_diagnostics.load_motion_windows can read the output.")
    parser.add_argument("--report-json", type=Path, default=None, help="Optional conversion report JSON path.")
    return parser


def _wants_isaaclab_backend(argv: list[str]) -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fk-backend", choices=("urdf", "isaaclab"), default="urdf")
    args, _ = parser.parse_known_args(argv)
    return args.fk_backend == "isaaclab"


def main() -> None:
    parser = _build_argparser()
    simulation_app = None
    if _wants_isaaclab_backend(sys.argv[1:]):
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        parser.set_defaults(headless=True)
        args = parser.parse_args()
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
    else:
        args, _ = parser.parse_known_args()
    try:
        target_fps = None if args.no_resample else args.target_fps
        device = getattr(args, "device", "cpu")
        if args.input_csv is not None:
            if args.output is None:
                raise ValueError("--output is required when using --input-csv")
            summaries = [
                convert_single_csv_to_smp_dataset(
                    csv_path=args.input_csv,
                    output_path=args.output,
                    source_fps=args.source_fps,
                    target_fps=target_fps,
                    window_size=args.window_size,
                    stride=args.stride,
                    fk_batch_size=args.fk_batch_size,
                    device=device,
                    fk_backend=args.fk_backend,
                    usd_path=args.usd_path,
                    urdf_path=args.urdf_path,
                    style_name=args.style_name,
                    style_id=args.style_id,
                    source_name=args.source_name,
                    verify=args.verify,
                )
            ]
        else:
            if args.output_dir is None:
                raise ValueError("--output-dir is required when using --input-dir")
            summaries = convert_csv_directory_to_smp_datasets(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                source_fps=args.source_fps,
                target_fps=target_fps,
                window_size=args.window_size,
                stride=args.stride,
                fk_batch_size=args.fk_batch_size,
                device=device,
                fk_backend=args.fk_backend,
                usd_path=args.usd_path,
                urdf_path=args.urdf_path,
                style_name=args.style_name,
                style_id=args.style_id,
                verify=args.verify,
            )
        _write_report(args.report_json, summaries)
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
