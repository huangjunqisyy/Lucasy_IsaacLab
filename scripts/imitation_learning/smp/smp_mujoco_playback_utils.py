#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence
import xml.etree.ElementTree as ET

import numpy as np
import torch


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONVERT_MODULE = _load_module(
    "isaaclab_smp_mujoco_playback_convert",
    _THIS_DIR / "convert_g1_csv_to_smp_dataset.py",
)
_SMP_FEATURES_MODULE = _load_module(
    "isaaclab_smp_mujoco_playback_features",
    _REPO_ROOT
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "mdp"
    / "smp_features.py",
)

FEATURE_DIM = int(_CONVERT_MODULE.g1_smp_feature_dim)
DATASET_JOINT_NAMES = list(_CONVERT_MODULE.g1_smp_joint_names)
DATASET_JOINT_AXES = np.asarray(_CONVERT_MODULE.g1_smp_joint_axes, dtype=np.float32)
DEFAULT_JOINT_POS = _CONVERT_MODULE._build_default_joint_pos(DATASET_JOINT_NAMES).cpu().numpy().astype(np.float32)
JOINT_ROT6D_TO_ANGLE_OFFSETS = _SMP_FEATURES_MODULE.joint_rot6d_to_angle_offsets


@dataclass(frozen=True)
class Trajectory:
    name: str
    fps: float
    root_pos: np.ndarray
    root_quat_xyzw: np.ndarray
    joint_pos: np.ndarray
    source_mode: str


def _scalar_to_float(value: np.ndarray | float | int | None, default: float) -> float:
    if value is None:
        return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    norms = np.linalg.norm(quat_xyzw, axis=-1, keepdims=True)
    norms = np.clip(norms, 1.0e-8, None)
    return (quat_xyzw / norms).astype(np.float32)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = _normalize_quat_xyzw(quat_xyzw)
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., 0:3]], axis=-1).astype(np.float32)


def _yaw_to_quat_xyzw(yaw_values: np.ndarray) -> np.ndarray:
    half = 0.5 * np.asarray(yaw_values, dtype=np.float64)
    quat = np.zeros((half.shape[0], 4), dtype=np.float64)
    quat[:, 2] = np.sin(half)
    quat[:, 3] = np.cos(half)
    return quat.astype(np.float32)


def _heading_rotation_from_yaw(yaw: float) -> np.ndarray:
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))
    return np.array(
        [
            [cos_yaw, 0.0, sin_yaw],
            [sin_yaw, 0.0, -cos_yaw],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def integrate_root_from_features(features: np.ndarray, fps: float, default_height: float) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] < 6:
        raise ValueError(f"Expected features shape (T, >=6), got {tuple(features.shape)}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")

    dt = 1.0 / float(fps)
    root_vel_local = features[:, 0:3]
    ang_vel_heading = features[:, 3:6]

    root_pos = np.zeros((features.shape[0], 3), dtype=np.float32)
    root_pos[:, 2] = float(default_height)
    yaw_values = np.zeros((features.shape[0],), dtype=np.float32)

    current_yaw = 0.0
    for frame_idx in range(1, features.shape[0]):
        heading_rotation = _heading_rotation_from_yaw(current_yaw)
        world_vel = heading_rotation @ root_vel_local[frame_idx - 1]
        root_pos[frame_idx] = root_pos[frame_idx - 1] + world_vel * dt
        root_pos[frame_idx, 2] = float(default_height)
        current_yaw += float(ang_vel_heading[frame_idx - 1, 1]) * dt
        yaw_values[frame_idx] = current_yaw

    return root_pos.astype(np.float32), _yaw_to_quat_xyzw(yaw_values)


def _decode_joint_positions_from_features(features: np.ndarray) -> np.ndarray:
    feature_tensor = torch.as_tensor(features, dtype=torch.float32)
    joint_rot6d = feature_tensor[..., 6:180].reshape(*feature_tensor.shape[:-1], len(DATASET_JOINT_NAMES), 6)
    joint_axes = torch.as_tensor(DATASET_JOINT_AXES, dtype=feature_tensor.dtype, device=feature_tensor.device)
    default_joint_pos = torch.as_tensor(DEFAULT_JOINT_POS, dtype=feature_tensor.dtype, device=feature_tensor.device)
    joint_offsets = JOINT_ROT6D_TO_ANGLE_OFFSETS(joint_rot6d, joint_axes)
    return (joint_offsets + default_joint_pos).cpu().numpy().astype(np.float32)


def _select_feature_sequence(
    data: np.lib.npyio.NpzFile,
    *,
    mode: str,
    sample_index: int,
    window_index: int,
) -> tuple[np.ndarray, float, str]:
    if mode in ("continuous", "auto") and "continuous_sequence_denormalized" in data:
        fps = _scalar_to_float(data["feature_fps"] if "feature_fps" in data else None, 50.0)
        return np.asarray(data["continuous_sequence_denormalized"], dtype=np.float32), fps, "continuous_sequence_denormalized"

    if mode in ("features", "auto") and "features" in data:
        fps = _scalar_to_float(data["feature_fps"] if "feature_fps" in data else data["target_fps"] if "target_fps" in data else None, 50.0)
        return np.asarray(data["features"], dtype=np.float32), fps, "features"

    if mode in ("frames", "auto") and "frames" in data:
        fps = _scalar_to_float(data["fps"] if "fps" in data else data["feature_fps"] if "feature_fps" in data else None, 50.0)
        return np.asarray(data["frames"], dtype=np.float32), fps, "frames"

    if mode in ("sample", "auto") and "samples_denormalized" in data:
        samples = np.asarray(data["samples_denormalized"], dtype=np.float32)
        if sample_index < 0 or sample_index >= samples.shape[0]:
            raise IndexError(f"sample_index={sample_index} 超出范围 [0, {samples.shape[0] - 1}]")
        fps = _scalar_to_float(data["feature_fps"] if "feature_fps" in data else None, 50.0)
        return samples[sample_index], fps, f"samples_denormalized[{sample_index}]"

    if mode in ("window", "auto") and "windows" in data:
        windows = np.asarray(data["windows"], dtype=np.float32)
        if windows.ndim == 4:
            windows = windows.reshape(-1, windows.shape[-2], windows.shape[-1])
        if window_index < 0 or window_index >= windows.shape[0]:
            raise IndexError(f"window_index={window_index} 超出范围 [0, {windows.shape[0] - 1}]")
        fps = _scalar_to_float(
            data["feature_fps"] if "feature_fps" in data else data["target_fps"] if "target_fps" in data else data["fps"] if "fps" in data else None,
            50.0,
        )
        return windows[window_index], fps, f"windows[{window_index}]"

    raise ValueError(f"无法从 NPZ 中找到适合 mode={mode!r} 的可视化内容")


def load_csv_trajectory(csv_path: str | Path, fps: float) -> Trajectory:
    motion = _CONVERT_MODULE.load_g1_csv_motion(csv_path, fps=float(fps))
    return Trajectory(
        name=Path(csv_path).name,
        fps=float(fps),
        root_pos=np.asarray(motion.root_pos, dtype=np.float32),
        root_quat_xyzw=_normalize_quat_xyzw(motion.root_quat_xyzw),
        joint_pos=np.asarray(motion.joint_pos, dtype=np.float32),
        source_mode="csv_raw",
    )


def decode_features_to_trajectory(
    *,
    sequence_name: str,
    features: np.ndarray,
    fps: float,
    source_mode: str,
    extractor=None,
    default_height: float,
) -> Trajectory:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected features shape (T, {FEATURE_DIM}), got {tuple(features.shape)}")

    if extractor is not None and hasattr(extractor, "decode_joint_positions"):
        feature_tensor = torch.as_tensor(features[None, ...], dtype=torch.float32)
        joint_pos = extractor.decode_joint_positions(feature_tensor).squeeze(0).detach().cpu().numpy().astype(np.float32)
    else:
        joint_pos = _decode_joint_positions_from_features(features)

    root_pos, root_quat_xyzw = integrate_root_from_features(features, fps=fps, default_height=default_height)
    return Trajectory(
        name=str(sequence_name),
        fps=float(fps),
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        source_mode=str(source_mode),
    )


def load_npz_trajectory(
    *,
    npz_path: str | Path,
    mode: str,
    sample_index: int,
    window_index: int,
    extractor=None,
    default_height: float,
    fps_override: float | None,
) -> Trajectory:
    npz_path = Path(npz_path)
    with np.load(npz_path) as data:
        has_pose_fields = {"root_pos", "root_quat_xyzw"} <= set(data.files) and ("joint_pos" in data.files or "joint_angles" in data.files)
        if mode in ("auto", "raw") and has_pose_fields:
            fps = fps_override or _scalar_to_float(data["source_fps"] if "source_fps" in data else data["fps"] if "fps" in data else None, 100.0)
            joint_key = "joint_pos" if "joint_pos" in data.files else "joint_angles"
            return Trajectory(
                name=npz_path.name,
                fps=float(fps),
                root_pos=np.asarray(data["root_pos"], dtype=np.float32),
                root_quat_xyzw=_normalize_quat_xyzw(np.asarray(data["root_quat_xyzw"], dtype=np.float32)),
                joint_pos=np.asarray(data[joint_key], dtype=np.float32),
                source_mode="raw_npz_fields",
            )

        features, fps, source_mode = _select_feature_sequence(
            data,
            mode=str(mode),
            sample_index=int(sample_index),
            window_index=int(window_index),
        )
        if fps_override is not None:
            fps = float(fps_override)
    return decode_features_to_trajectory(
        sequence_name=npz_path.name,
        features=features,
        fps=fps,
        source_mode=source_mode,
        extractor=extractor,
        default_height=default_height,
    )


def collect_input_files(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        files = sorted(list(input_path.glob("*.csv")) + list(input_path.glob("*.npz")))
        if not files:
            raise ValueError(f"{input_path} 中没有找到 csv 或 npz 文件")
        return files
    raise ValueError(f"输入路径不存在: {input_path}")


def load_pose_arrays_from_csv(csv_path: str | Path, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    motion = _CONVERT_MODULE.load_g1_csv_motion(csv_path, fps=float(fps))
    return (
        np.asarray(motion.root_pos, dtype=np.float32),
        _normalize_quat_xyzw(motion.root_quat_xyzw),
        np.asarray(motion.joint_pos, dtype=np.float32),
        float(motion.fps),
    )


def load_pose_arrays_from_npz(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(npz_path) as data:
        has_pose_fields = {"root_pos", "root_quat_xyzw"} <= set(data.files) and ("joint_pos" in data.files or "joint_angles" in data.files)
        if not has_pose_fields:
            raise ValueError(f"{npz_path} does not contain pose fields")
        joint_key = "joint_pos" if "joint_pos" in data.files else "joint_angles"
        fps = _scalar_to_float(data["source_fps"] if "source_fps" in data else data["fps"] if "fps" in data else None, 100.0)
        return (
            np.asarray(data["root_pos"], dtype=np.float32),
            _normalize_quat_xyzw(np.asarray(data["root_quat_xyzw"], dtype=np.float32)),
            np.asarray(data[joint_key], dtype=np.float32),
            float(fps),
        )


def build_frames_from_pose_arrays(
    *,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_pos: np.ndarray,
    fps: float,
    urdf_path: str | Path | None = None,
) -> np.ndarray:
    fk = _CONVERT_MODULE.G1UrdfForwardKinematics(urdf_path=urdf_path)
    ee_pos_w = fk.compute_ee_positions(
        root_pos=np.asarray(root_pos, dtype=np.float32),
        root_quat_xyzw=np.asarray(root_quat_xyzw, dtype=np.float32),
        joint_pos=np.asarray(joint_pos, dtype=np.float32),
    )
    return _CONVERT_MODULE.build_smp_frames_from_motion_arrays(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        ee_pos_w=ee_pos_w,
        fps=float(fps),
    )


def build_sliding_windows(frames: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"Expected frames shape (T, F), got {tuple(frames.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if frames.shape[0] < window_size:
        raise ValueError(f"Expected at least {window_size} frames, got {frames.shape[0]}")
    windows = [frames[start : start + window_size] for start in range(0, frames.shape[0] - window_size + 1, stride)]
    return np.stack(windows, axis=0).astype(np.float32)


def select_window_batch(
    sequence_or_windows: np.ndarray,
    *,
    window_size: int,
    stride: int,
    window_index: int,
    num_windows: int,
) -> np.ndarray:
    sequence_or_windows = np.asarray(sequence_or_windows, dtype=np.float32)
    if num_windows <= 0:
        raise ValueError(f"num_windows must be positive, got {num_windows}")

    if sequence_or_windows.ndim == 3:
        end_index = int(window_index) + int(num_windows)
        if window_index < 0 or end_index > sequence_or_windows.shape[0]:
            raise IndexError(
                f"window range [{window_index}, {end_index}) 超出范围 [0, {sequence_or_windows.shape[0]})"
            )
        return sequence_or_windows[int(window_index) : end_index].astype(np.float32)

    if sequence_or_windows.ndim != 2:
        raise ValueError(f"Expected sequence shape (T, F) or windows shape (N, W, F), got {tuple(sequence_or_windows.shape)}")

    if sequence_or_windows.shape[0] == window_size:
        if window_index != 0 or num_windows != 1:
            raise IndexError("Selected sequence already matches one window; window_index must be 0 and num_windows must be 1")
        return sequence_or_windows[None, ...].astype(np.float32)

    windows = build_sliding_windows(sequence_or_windows, window_size=window_size, stride=stride)
    end_index = int(window_index) + int(num_windows)
    if window_index < 0 or end_index > windows.shape[0]:
        raise IndexError(f"window range [{window_index}, {end_index}) 超出范围 [0, {windows.shape[0]})")
    return windows[int(window_index) : end_index].astype(np.float32)


def _format_window_source_mode(*, prefix: str, window_index: int, num_windows: int) -> str:
    end_index = int(window_index) + int(num_windows)
    if int(num_windows) == 1:
        return f"{prefix}[{int(window_index)}]"
    return f"{prefix}[{int(window_index)}:{end_index}]"


def load_feature_window_batch_from_input(
    *,
    input_path: str | Path,
    window_size: int,
    stride: int = 1,
    mode: str = "auto",
    csv_fps: float = 30.0,
    sample_index: int = 0,
    window_index: int = 0,
    num_windows: int = 1,
    urdf_path: str | Path | None = None,
) -> tuple[np.ndarray, float, str]:
    input_path = Path(input_path).expanduser().resolve()
    if input_path.suffix.lower() == ".csv":
        root_pos, root_quat_xyzw, joint_pos, fps = load_pose_arrays_from_csv(input_path, fps=csv_fps)
        frames = build_frames_from_pose_arrays(
            root_pos=root_pos,
            root_quat_xyzw=root_quat_xyzw,
            joint_pos=joint_pos,
            fps=fps,
            urdf_path=urdf_path,
        )
        return (
            select_window_batch(
                frames,
                window_size=window_size,
                stride=stride,
                window_index=window_index,
                num_windows=num_windows,
            ),
            float(fps),
            "csv_frames",
        )

    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported input suffix: {input_path.suffix}")

    with np.load(input_path) as data:
        if mode != "raw":
            if mode in ("window", "auto") and "windows" in data:
                fps = _scalar_to_float(
                    data["feature_fps"] if "feature_fps" in data else data["target_fps"] if "target_fps" in data else data["fps"] if "fps" in data else None,
                    50.0,
                )
                windows = np.asarray(data["windows"], dtype=np.float32)
                if windows.ndim == 4:
                    windows = windows.reshape(-1, windows.shape[-2], windows.shape[-1])
                return (
                    select_window_batch(
                        windows,
                        window_size=window_size,
                        stride=stride,
                        window_index=window_index,
                        num_windows=num_windows,
                    ),
                    float(fps),
                    _format_window_source_mode(prefix="windows", window_index=window_index, num_windows=num_windows),
                )

            try:
                sequence, fps, source_mode = _select_feature_sequence(
                    data,
                    mode=str(mode),
                    sample_index=int(sample_index),
                    window_index=int(window_index),
                )
                return (
                    select_window_batch(
                        sequence,
                        window_size=window_size,
                        stride=stride,
                        window_index=window_index,
                        num_windows=num_windows,
                    ),
                    float(fps),
                    source_mode,
                )
            except ValueError:
                if mode != "auto":
                    raise

        has_pose_fields = {"root_pos", "root_quat_xyzw"} <= set(data.files) and ("joint_pos" in data.files or "joint_angles" in data.files)
        if not has_pose_fields:
            raise ValueError(f"{input_path} does not contain pose fields")
        joint_key = "joint_pos" if "joint_pos" in data.files else "joint_angles"
        fps = _scalar_to_float(data["source_fps"] if "source_fps" in data else data["fps"] if "fps" in data else None, 100.0)
        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_quat_xyzw = _normalize_quat_xyzw(np.asarray(data["root_quat_xyzw"], dtype=np.float32))
        joint_pos = np.asarray(data[joint_key], dtype=np.float32)

    frames = build_frames_from_pose_arrays(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        fps=fps,
        urdf_path=urdf_path,
    )
    return (
        select_window_batch(
            frames,
            window_size=window_size,
            stride=stride,
            window_index=window_index,
            num_windows=num_windows,
        ),
        float(fps),
        "raw_pose_frames",
    )


def load_feature_sequence_from_input(
    *,
    input_path: str | Path,
    mode: str = "auto",
    csv_fps: float = 30.0,
    sample_index: int = 0,
    window_index: int = 0,
    urdf_path: str | Path | None = None,
) -> tuple[np.ndarray, float, str]:
    input_path = Path(input_path).expanduser().resolve()
    if input_path.suffix.lower() == ".csv":
        root_pos, root_quat_xyzw, joint_pos, fps = load_pose_arrays_from_csv(input_path, fps=csv_fps)
        frames = build_frames_from_pose_arrays(
            root_pos=root_pos,
            root_quat_xyzw=root_quat_xyzw,
            joint_pos=joint_pos,
            fps=fps,
            urdf_path=urdf_path,
        )
        return frames, float(fps), "csv_frames"

    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported input suffix: {input_path.suffix}")

    with np.load(input_path) as data:
        if mode != "raw":
            try:
                sequence, fps, source_mode = _select_feature_sequence(
                    data,
                    mode=str(mode),
                    sample_index=int(sample_index),
                    window_index=int(window_index),
                )
                return np.asarray(sequence, dtype=np.float32), float(fps), source_mode
            except ValueError:
                if mode != "auto":
                    raise

        has_pose_fields = {"root_pos", "root_quat_xyzw"} <= set(data.files) and ("joint_pos" in data.files or "joint_angles" in data.files)
        if has_pose_fields:
            root_pos, root_quat_xyzw, joint_pos, fps = load_pose_arrays_from_npz(input_path)
            if "ee_pos_w" in data.files:
                frames = _CONVERT_MODULE.build_smp_frames_from_motion_arrays(
                    root_pos=root_pos,
                    root_quat_xyzw=root_quat_xyzw,
                    joint_pos=joint_pos,
                    ee_pos_w=np.asarray(data["ee_pos_w"], dtype=np.float32),
                    fps=fps,
                )
            else:
                frames = build_frames_from_pose_arrays(
                    root_pos=root_pos,
                    root_quat_xyzw=root_quat_xyzw,
                    joint_pos=joint_pos,
                    fps=fps,
                    urdf_path=urdf_path,
                )
            return frames, float(fps), "raw_pose_npz_frames"
        raise ValueError(f"无法从 {input_path} 中解析出 mode={mode!r} 的特征序列")


def print_trajectory_summary(trajectory: Trajectory) -> None:
    duration = len(trajectory.joint_pos) / float(trajectory.fps)
    print(
        f"[{trajectory.name}] mode={trajectory.source_mode} "
        f"frames={len(trajectory.joint_pos)} fps={trajectory.fps:.2f} duration={duration:.2f}s "
        f"root_z=[{trajectory.root_pos[:, 2].min():.3f}, {trajectory.root_pos[:, 2].max():.3f}]"
    )


def resolve_g1_urdf_path(explicit_path: str | Path | None = None) -> Path:
    return Path(_CONVERT_MODULE._resolve_g1_urdf_path(explicit_path)).resolve()


def build_floating_model_from_urdf(urdf_path: str | Path):
    import mujoco

    urdf_path = Path(urdf_path).expanduser().resolve()
    base_model = mujoco.MjModel.from_xml_path(str(urdf_path))

    with tempfile.TemporaryDirectory(prefix="g1_mjcf_") as tmp_dir:
        saved_xml = Path(tmp_dir) / "base.xml"
        floating_xml = Path(tmp_dir) / "floating.xml"
        mujoco.mj_saveLastXML(str(saved_xml), base_model)

        tree = ET.parse(saved_xml)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        compiler.set("meshdir", str(urdf_path.parent / "meshes") + "/")

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("生成的 MJCF 缺少 worldbody")

        children = list(worldbody)
        for child in children:
            worldbody.remove(child)

        floating_base = ET.SubElement(worldbody, "body", {"name": "floating_base", "pos": "0 0 0"})
        ET.SubElement(floating_base, "freejoint", {"name": "root_freejoint"})
        for child in children:
            floating_base.append(child)

        tree.write(floating_xml, encoding="unicode")
        return mujoco.MjModel.from_xml_path(str(floating_xml))


def _joint_qpos_indices(model) -> np.ndarray:
    import mujoco

    indices = []
    for joint_name in DATASET_JOINT_NAMES:
        candidate_names = [str(joint_name)]
        if not str(joint_name).endswith("_joint"):
            candidate_names.append(f"{joint_name}_joint")

        joint_id = -1
        resolved_name = candidate_names[0]
        for candidate_name in candidate_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, candidate_name)
            if joint_id >= 0:
                resolved_name = candidate_name
                break
        if joint_id < 0:
            raise ValueError(f"MuJoCo 模型中找不到关节 {candidate_names}")
        indices.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(indices, dtype=np.int32)


def _apply_frame_to_data(data, joint_indices: np.ndarray, root_pos: np.ndarray, root_quat_xyzw: np.ndarray, joint_pos: np.ndarray) -> None:
    import mujoco

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = _xyzw_to_wxyz(root_quat_xyzw)
    data.qpos[joint_indices] = joint_pos
    mujoco.mj_forward(data.model, data)


def play_trajectories(model, trajectories: Sequence[Trajectory], loop: bool = False, hold: float = 0.5) -> None:
    import mujoco
    import mujoco.viewer

    data = mujoco.MjData(model)
    joint_indices = _joint_qpos_indices(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        while viewer.is_running():
            for trajectory in trajectories:
                print_trajectory_summary(trajectory)
                dt = 1.0 / float(trajectory.fps)
                for frame_idx in range(len(trajectory.joint_pos)):
                    frame_start = time.perf_counter()
                    _apply_frame_to_data(
                        data=data,
                        joint_indices=joint_indices,
                        root_pos=trajectory.root_pos[frame_idx],
                        root_quat_xyzw=trajectory.root_quat_xyzw[frame_idx],
                        joint_pos=trajectory.joint_pos[frame_idx],
                    )
                    viewer.sync()
                    elapsed = time.perf_counter() - frame_start
                    if elapsed < dt:
                        time.sleep(dt - elapsed)
                    if not viewer.is_running():
                        return
                if hold > 0.0:
                    time.sleep(float(hold))
            if not loop:
                break
