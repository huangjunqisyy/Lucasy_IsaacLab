# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """将四元数转换为 6D 旋转表示。"""
    mat = math_utils.matrix_from_quat(quat_wxyz.reshape(-1, 4)).reshape(*quat_wxyz.shape[:-1], 3, 3)
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
    key_body_rot6d = quat_to_rot6d(key_body_quat_b).reshape(key_body_quat_b.shape[0], -1)
    ee_pos_b = ee_pos_b.reshape(ee_pos_b.shape[0], -1)
    features = torch.cat([base_lin_vel_b, base_ang_vel_b, joint_pos_rel, ee_pos_b, key_body_rot6d], dim=-1)
    # 使用显式维度检查，避免后续训练阶段静默出现特征长度漂移。
    if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
        raise ValueError(f"Expected SMP feature dim {expected_feature_dim}, got {features.shape[-1]}")
    return features
