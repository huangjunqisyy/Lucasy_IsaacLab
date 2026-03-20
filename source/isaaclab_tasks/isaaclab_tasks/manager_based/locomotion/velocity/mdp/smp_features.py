# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def _matrix_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """使用纯 torch 将四元数转换为旋转矩阵。"""
    quat_wxyz = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quat_wxyz.unbind(dim=-1)

    return torch.stack(
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
        dim=-1,
    ).reshape(*quat_wxyz.shape[:-1], 3, 3)


def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """将四元数转换为 6D 旋转表示。

    这里固定采用“旋转矩阵前两列，按默认行优先顺序展平”的约定。
    例如单位四元数会得到 `[1, 0, 0, 1, 0, 0]`。
    """
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quat_wxyz last dim 4, got {quat_wxyz.shape[-1]}")
    # 这里保持纯 torch 依赖，避免单元测试被 Isaac/Omni 导入链污染。
    mat = _matrix_from_quat(quat_wxyz)
    return mat[..., :2].reshape(*quat_wxyz.shape[:-1], 6)


def pack_smp_frame_features(
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    ee_pos_b: torch.Tensor,
    key_body_quat_b: torch.Tensor,
    expected_feature_dim: int | None = None,
) -> torch.Tensor:
    """将单帧 SMP 特征按照统一顺序拼接成向量。"""
    lead_shape = base_lin_vel_b.shape[:-1]
    if base_lin_vel_b.shape[-1] != 3:
        raise ValueError(f"Expected base_lin_vel_b last dim 3, got {base_lin_vel_b.shape[-1]}")
    if base_ang_vel_b.shape[-1] != 3:
        raise ValueError(f"Expected base_ang_vel_b last dim 3, got {base_ang_vel_b.shape[-1]}")
    if base_ang_vel_b.shape[:-1] != lead_shape or joint_pos_rel.shape[:-1] != lead_shape:
        raise ValueError("SMP 特征输入的前导维度不一致")
    if ee_pos_b.shape[:-2] != lead_shape or key_body_quat_b.shape[:-2] != lead_shape:
        raise ValueError("SMP 末端执行器或关键刚体输入的前导维度不一致")
    if ee_pos_b.shape[-1] != 3:
        raise ValueError(f"Expected ee_pos_b last dim 3, got {ee_pos_b.shape[-1]}")
    if key_body_quat_b.shape[-1] != 4:
        raise ValueError(f"Expected key_body_quat_b last dim 4, got {key_body_quat_b.shape[-1]}")

    key_body_rot6d = quat_to_rot6d(key_body_quat_b).reshape(*lead_shape, -1)
    ee_pos_b = ee_pos_b.reshape(*lead_shape, -1)
    features = torch.cat([base_lin_vel_b, base_ang_vel_b, joint_pos_rel, ee_pos_b, key_body_rot6d], dim=-1)
    # 使用显式维度检查，避免后续训练阶段静默出现特征长度漂移。
    if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
        raise ValueError(f"Expected SMP feature dim {expected_feature_dim}, got {features.shape[-1]}")
    return features
