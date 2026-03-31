# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import torch


def _load_module(module_name: str, module_path: Path):
    # 通过文件路径加载模块，避免依赖 PYTHONPATH 或包安装顺序。
    # 这里用于复用 IsaacLab 中已定义的特征打包函数和 G1 配置常量。
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SMP_FEATURES_MODULE = _load_module(
    "isaaclab_smp_export_features",
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
_G1_CONFIG_MODULE = _load_module(
    "isaaclab_smp_export_g1_config",
    _REPO_ROOT
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "locomotion"
    / "velocity"
    / "config"
    / "g1"
    / "agents"
    / "config.py",
)

# 从已加载模块中提取运行时所需的“单一真值”配置，
# 让数据导出脚本与训练配置保持一致，避免手动复制参数导致偏差。
pack_smp_frame_features = _SMP_FEATURES_MODULE.pack_smp_frame_features
g1_ee_names = list(_G1_CONFIG_MODULE.g1_ee_names)
g1_key_body_names = list(_G1_CONFIG_MODULE.g1_key_body_names)
g1_smp_feature_dim = int(_G1_CONFIG_MODULE.g1_smp_feature_dim)
g1_smp_joint_names = list(_G1_CONFIG_MODULE.g1_smp_joint_names)
g1_smp_window_size = int(_G1_CONFIG_MODULE.g1_smp_window_size)

# 原始 motion 文件中的 body 维度固定顺序。
# 下游通过名称映射拿索引，因此这里的顺序必须与数据源严格一致。
_G1_MOTION_BODY_ORDER = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]
_G1_BODY_NAME_TO_INDEX = {name: idx for idx, name in enumerate(_G1_MOTION_BODY_ORDER)}

# 构造“默认关节位姿”时使用的规则。
# 这些值通常对应机器人自然站立姿态，用于把绝对关节角转成相对偏移（joint_pos_rel）。
_G1_DEFAULT_JOINT_POS_RULES = (
    ("left_hip_pitch_joint", -0.1),
    ("right_hip_pitch_joint", -0.1),
    (re.compile(r".*_knee_joint"), 0.3),
    (re.compile(r".*_ankle_pitch_joint"), -0.2),
    (re.compile(r".*_shoulder_pitch_joint"), 0.3),
    ("left_shoulder_roll_joint", 0.25),
    ("right_shoulder_roll_joint", -0.25),
    (re.compile(r".*_elbow_joint"), 0.97),
    ("left_wrist_roll_joint", 0.15),
    ("right_wrist_roll_joint", -0.15),
)


def _quat_conjugate(quat_wxyz: torch.Tensor) -> torch.Tensor:
    # 四元数共轭: q* = [w, -x, -y, -z]
    # 在单位四元数下，q* 等价于 q^{-1}。
    return torch.cat((quat_wxyz[..., :1], -quat_wxyz[..., 1:]), dim=-1)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    # Hamilton 乘法，输入输出均为 wxyz 排列。
    # 支持批量维度，最后一维为 4。
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_apply(quat_wxyz: torch.Tensor, vec_xyz: torch.Tensor) -> torch.Tensor:
    # 将向量 vec 旋转到 quat 对应姿态的坐标系。
    # 使用向量形式实现（叉乘），避免显式构造 [0, v] 四元数。
    quat_xyz = quat_wxyz[..., 1:]
    uv = torch.cross(quat_xyz, vec_xyz, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vec_xyz + 2.0 * (quat_wxyz[..., :1] * uv + uuv)


def _quat_apply_inverse(quat_wxyz: torch.Tensor, vec_xyz: torch.Tensor) -> torch.Tensor:
    # 使用 q^{-1} 进行旋转，等价于把世界系向量变换到 body 系。
    return _quat_apply(_quat_conjugate(quat_wxyz), vec_xyz)


def _build_default_joint_pos(joint_names: list[str]) -> torch.Tensor:
    # 根据关节名匹配规则生成默认位姿向量，未匹配到的关节默认 0。
    # 输出 shape: (num_joints,)
    default_joint_pos = []
    for joint_name in joint_names:
        joint_value = 0.0
        for pattern, value in _G1_DEFAULT_JOINT_POS_RULES:
            if isinstance(pattern, str) and joint_name == pattern:
                joint_value = value
                break
            if hasattr(pattern, "fullmatch") and pattern.fullmatch(joint_name):
                joint_value = value
                break
        default_joint_pos.append(joint_value)
    return torch.tensor(default_joint_pos, dtype=torch.float32)


def _load_motion_arrays(input_path: Path) -> tuple[dict[str, np.ndarray], float]:
    # 读取原始 npz，并把所有字段转为 ndarray。
    # fps 单独抽出为 float，便于后续保存 metadata。
    with np.load(input_path) as data:
        if "fps" not in data:
            raise KeyError("Input motion npz must contain 'fps'")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return arrays, fps


def _build_smp_frames_from_raw_motion(raw_motion: dict[str, np.ndarray]) -> torch.Tensor:
    # 从原始运动字段构造每一帧 SMP 特征。
    # 输入字段期望 shape:
    # - joint_pos:      (T, J)
    # - body_pos_w:     (T, B, 3)
    # - body_quat_w:    (T, B, 4)  [w, x, y, z]
    # - body_lin_vel_w: (T, B, 3)
    # - body_ang_vel_w: (T, B, 3)
    required_keys = ("joint_pos", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
    missing_keys = [key for key in required_keys if key not in raw_motion]
    if missing_keys:
        raise KeyError(f"Input motion npz is missing keys: {missing_keys}")

    joint_pos = torch.as_tensor(raw_motion["joint_pos"], dtype=torch.float32)
    body_pos_w = torch.as_tensor(raw_motion["body_pos_w"], dtype=torch.float32)
    body_quat_w = torch.as_tensor(raw_motion["body_quat_w"], dtype=torch.float32)
    body_lin_vel_w = torch.as_tensor(raw_motion["body_lin_vel_w"], dtype=torch.float32)
    body_ang_vel_w = torch.as_tensor(raw_motion["body_ang_vel_w"], dtype=torch.float32)

    if joint_pos.ndim != 2:
        raise ValueError(f"Expected joint_pos shape (num_frames, num_joints), got {joint_pos.shape}")
    if joint_pos.shape[-1] != len(g1_smp_joint_names):
        raise ValueError(f"Expected {len(g1_smp_joint_names)} joints, got {joint_pos.shape[-1]}")
    if body_pos_w.shape[1] != len(_G1_MOTION_BODY_ORDER):
        raise ValueError(f"Expected {len(_G1_MOTION_BODY_ORDER)} bodies, got {body_pos_w.shape[1]}")

    # 通过名称映射索引，保证即使索引常量变动也能按语义取数据。
    root_body_index = _G1_BODY_NAME_TO_INDEX["pelvis"]
    ee_body_ids = [_G1_BODY_NAME_TO_INDEX[name] for name in g1_ee_names]
    key_body_ids = [_G1_BODY_NAME_TO_INDEX[name] for name in g1_key_body_names]
    default_joint_pos = _build_default_joint_pos(g1_smp_joint_names).to(joint_pos.device)

    # Root 基座特征: 世界系速度 -> 基座系速度。
    root_pos_w = body_pos_w[:, root_body_index]
    root_quat_w = body_quat_w[:, root_body_index]
    base_lin_vel_b = _quat_apply_inverse(root_quat_w, body_lin_vel_w[:, root_body_index])
    base_ang_vel_b = _quat_apply_inverse(root_quat_w, body_ang_vel_w[:, root_body_index])

    # 关节特征: 使用相对默认姿态的偏移，减少静态姿态偏置。
    joint_pos_rel = joint_pos - default_joint_pos.unsqueeze(0)

    # EE 位置特征: 先做平移到 root 原点，再旋转到 root/body 坐标系。
    ee_pos_w = body_pos_w[:, ee_body_ids]
    ee_pos_b = _quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, len(ee_body_ids), -1),
        ee_pos_w - root_pos_w.unsqueeze(1),
    )

    # 关键 body 朝向特征: q_key_in_root = q_root^{-1} * q_key。
    key_body_quat_w = body_quat_w[:, key_body_ids]
    key_body_quat_b = _quat_mul(
        _quat_conjugate(root_quat_w).unsqueeze(1).expand(-1, len(key_body_ids), -1),
        key_body_quat_w,
    )

    # 统一由 pack 函数按固定顺序拼接特征，并校验 feature 维度。
    return pack_smp_frame_features(
        base_lin_vel_b=base_lin_vel_b,
        base_ang_vel_b=base_ang_vel_b,
        joint_pos_rel=joint_pos_rel,
        ee_pos_b=ee_pos_b,
        key_body_quat_b=key_body_quat_b,
        expected_feature_dim=g1_smp_feature_dim,
    )


def export_g1_motion_dataset(
    input_path: str | Path,
    output_path: str | Path,
    window_size: int = g1_smp_window_size,
    stride: int = 1,
    style_name: str | None = None,
    style_id: int | None = None,
    source_name: str | None = None,
) -> Path:
    """将 G1 运动文件导出为 SMP 帧特征数据集。

    行为说明:
    1. 读取输入 npz（至少包含 fps 和原始运动字段，或直接包含 frames）。
    2. 若存在 frames 字段则直接复用；否则由原始 body/joint 字段在线构造。
    3. 把帧特征与训练需要的 metadata 一并写入输出 npz。
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    raw_motion, fps = _load_motion_arrays(input_path)

    # 兼容两类输入:
    # - 已预处理: 直接包含 frames
    # - 原始 motion: 需要先转换为 SMP frame features
    if "frames" in raw_motion:
        frames = torch.as_tensor(raw_motion["frames"], dtype=torch.float32)
    else:
        frames = _build_smp_frames_from_raw_motion(raw_motion)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 除了帧数据，还保存 feature_dim/window_size/stride 及名称列表，
    # 便于训练和离线检查时做一致性校验。
    save_payload = {
        "frames": frames.cpu().numpy().astype(np.float32),
        "fps": np.array([fps], dtype=np.float32),
        "window_size": np.array([window_size], dtype=np.int64),
        "stride": np.array([stride], dtype=np.int64),
        "feature_dim": np.array([frames.shape[-1]], dtype=np.int64),
        "joint_names": np.asarray(g1_smp_joint_names),
        "ee_names": np.asarray(g1_ee_names),
        "key_body_names": np.asarray(g1_key_body_names),
    }
    if style_name is not None:
        save_payload["style_name"] = np.asarray([style_name], dtype=np.str_)
    if style_id is not None:
        save_payload["style_id"] = np.asarray([style_id], dtype=np.int64)
    if source_name is not None:
        save_payload["source_name"] = np.asarray([source_name], dtype=np.str_)
    np.savez(output_path, **save_payload)
    return output_path


def _build_argparser() -> argparse.ArgumentParser:
    # CLI 仅负责参数读取，业务逻辑统一收敛到 export_g1_motion_dataset。
    parser = argparse.ArgumentParser(description="导出 G1 SMP 运动先验训练数据集。")
    parser.add_argument("--input", required=True, help="输入的原始 G1 motion npz 路径。")
    parser.add_argument("--output", required=True, help="输出的 SMP 帧数据集 npz 路径。")
    parser.add_argument("--window-size", required=True, type=int, help="写入 metadata 的窗口大小。")
    parser.add_argument("--stride", type=int, default=1, help="写入 metadata 的滑窗步长。")
    parser.add_argument("--style-name", default=None, help="可选：当前数据片段对应的风格名。")
    parser.add_argument("--style-id", type=int, default=None, help="可选：当前数据片段对应的风格 id。")
    parser.add_argument("--source-name", default=None, help="可选：数据源名字，用于回溯 shard。")
    return parser


def main():
    # 脚本入口: 解析命令行并执行导出。
    args = _build_argparser().parse_args()
    export_g1_motion_dataset(
        input_path=args.input,
        output_path=args.output,
        window_size=args.window_size,
        stride=args.stride,
        style_name=args.style_name,
        style_id=args.style_id,
        source_name=args.source_name,
    )


if __name__ == "__main__":
    main()
