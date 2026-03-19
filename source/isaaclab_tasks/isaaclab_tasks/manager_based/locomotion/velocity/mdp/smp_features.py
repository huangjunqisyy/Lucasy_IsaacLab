# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils


def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to 6D rotation representations.

    Args:
        quat_wxyz: Quaternion tensor in (w, x, y, z) format with shape (..., 4).

    Returns:
        Tensor with shape (..., 6) containing the first two columns of the rotation matrix.
    """
    mat = math_utils.matrix_from_quat(quat_wxyz.reshape(-1, 4)).reshape(*quat_wxyz.shape[:-1], 3, 3)
    return mat[..., :2].reshape(*quat_wxyz.shape[:-1], 6)


def pack_smp_frame_features(
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    joint_pos_rel: torch.Tensor,
    ee_pos_b: torch.Tensor,
    key_body_quat_b: torch.Tensor,
) -> torch.Tensor:
    """Pack a single SMP frame feature vector from common tensors."""
    key_body_rot6d = quat_to_rot6d(key_body_quat_b).reshape(key_body_quat_b.shape[0], -1)
    ee_pos_b = ee_pos_b.reshape(ee_pos_b.shape[0], -1)
    return torch.cat([base_lin_vel_b, base_ang_vel_b, joint_pos_rel, ee_pos_b, key_body_rot6d], dim=-1)
