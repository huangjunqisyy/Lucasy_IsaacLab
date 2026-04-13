# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch


def _normalize(vec: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    return vec / vec.norm(dim=-1, keepdim=True).clamp_min(eps)


def _matrix_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """使用纯 torch 将四元数转换为旋转矩阵。"""
    quat_wxyz = _normalize(quat_wxyz)
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


def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """将行优先展开的前两列 6D 旋转表示还原为旋转矩阵。"""
    if rot6d.shape[-1] != 6:
        raise ValueError(f"Expected rot6d last dim 6, got {rot6d.shape[-1]}")

    col_1 = torch.stack((rot6d[..., 0], rot6d[..., 2], rot6d[..., 4]), dim=-1)
    col_2 = torch.stack((rot6d[..., 1], rot6d[..., 3], rot6d[..., 5]), dim=-1)
    basis_1 = _normalize(col_1)
    basis_2 = _normalize(col_2 - (basis_1 * col_2).sum(dim=-1, keepdim=True) * basis_1)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """将四元数转换为 6D 旋转表示。

    这里固定采用“旋转矩阵前两列，按默认行优先顺序展平”的约定。
    例如单位四元数会得到 `[1, 0, 0, 1, 0, 0]`。
    """
    if quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected quat_wxyz last dim 4, got {quat_wxyz.shape[-1]}")
    mat = _matrix_from_quat(quat_wxyz)
    return mat[..., :2].reshape(*quat_wxyz.shape[:-1], 6)


def build_heading_frame_rotation(root_quat_wxyz: torch.Tensor) -> torch.Tensor:
    """构造 pelvis-heading 局部坐标系到世界系的旋转矩阵。

    局部坐标系定义为:
    - x 轴: root link facing direction 在水平面的投影
    - y 轴: 全局 up 向量
    - z 轴: `x × y`
    """
    if root_quat_wxyz.shape[-1] != 4:
        raise ValueError(f"Expected root_quat_wxyz last dim 4, got {root_quat_wxyz.shape[-1]}")

    root_rotation = _matrix_from_quat(root_quat_wxyz)
    forward_w = root_rotation[..., :, 0]
    fallback_w = root_rotation[..., :, 1]
    up_w = torch.zeros_like(forward_w)
    up_w[..., 2] = 1.0

    forward_proj = forward_w - (forward_w * up_w).sum(dim=-1, keepdim=True) * up_w
    fallback_proj = fallback_w - (fallback_w * up_w).sum(dim=-1, keepdim=True) * up_w
    use_fallback = forward_proj.norm(dim=-1, keepdim=True) < 1.0e-6
    x_axis_w = _normalize(torch.where(use_fallback, fallback_proj, forward_proj))
    y_axis_w = up_w
    z_axis_w = _normalize(torch.cross(x_axis_w, y_axis_w, dim=-1))
    x_axis_w = _normalize(torch.cross(y_axis_w, z_axis_w, dim=-1))
    return torch.stack((x_axis_w, y_axis_w, z_axis_w), dim=-1)


def world_to_local_frame(rotation_world_from_local: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    """将世界系向量变换到给定局部坐标系。"""
    if rotation_world_from_local.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotation_world_from_local trailing shape (3, 3), got {rotation_world_from_local.shape[-2:]}"
        )
    if vec_w.shape[-1] != 3:
        raise ValueError(f"Expected vec_w last dim 3, got {vec_w.shape[-1]}")
    while rotation_world_from_local.ndim < vec_w.ndim + 1:
        rotation_world_from_local = rotation_world_from_local.unsqueeze(-3)
    return torch.matmul(vec_w.unsqueeze(-2), rotation_world_from_local).squeeze(-2)


def joint_angle_offsets_to_rot6d(joint_angle_offsets: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    """将单轴关节角偏移编码为连续 rot6d。"""
    if joint_angle_offsets.ndim < 1:
        raise ValueError("Expected joint_angle_offsets to have at least one dimension")
    if joint_axes.ndim != 2 or joint_axes.shape[-1] != 3:
        raise ValueError(f"Expected joint_axes shape (num_joints, 3), got {joint_axes.shape}")
    if joint_angle_offsets.shape[-1] != joint_axes.shape[0]:
        raise ValueError(
            f"Expected joint_angle_offsets last dim {joint_axes.shape[0]}, got {joint_angle_offsets.shape[-1]}"
        )

    normalized_axes = _normalize(joint_axes.to(device=joint_angle_offsets.device, dtype=joint_angle_offsets.dtype))
    axis_shape = (1,) * (joint_angle_offsets.ndim - 1) + normalized_axes.shape
    expanded_axes = normalized_axes.view(axis_shape)
    half_angle = 0.5 * joint_angle_offsets
    quat_wxyz = torch.cat(
        (
            torch.cos(half_angle).unsqueeze(-1),
            expanded_axes * torch.sin(half_angle).unsqueeze(-1),
        ),
        dim=-1,
    )
    return quat_to_rot6d(quat_wxyz)


def joint_positions_to_rot6d(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_axes: torch.Tensor,
) -> torch.Tensor:
    """将关节位置编码为“相对默认站姿”的 rot6d。"""
    if joint_pos.shape[-1] != joint_axes.shape[0]:
        raise ValueError(f"Expected joint_pos last dim {joint_axes.shape[0]}, got {joint_pos.shape[-1]}")
    return joint_angle_offsets_to_rot6d(joint_pos - default_joint_pos, joint_axes)


def joint_rot6d_to_angle_offsets(joint_rot6d: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    """将单轴关节的 rot6d 表示还原为关节角偏移。"""
    if joint_rot6d.shape[-1] != 6:
        raise ValueError(f"Expected joint_rot6d last dim 6, got {joint_rot6d.shape[-1]}")
    if joint_axes.ndim != 2 or joint_axes.shape[-1] != 3:
        raise ValueError(f"Expected joint_axes shape (num_joints, 3), got {joint_axes.shape}")
    if joint_rot6d.shape[-2] != joint_axes.shape[0]:
        raise ValueError(f"Expected joint_rot6d joint dim {joint_axes.shape[0]}, got {joint_rot6d.shape[-2]}")

    rotmat = _rot6d_to_matrix(joint_rot6d)
    normalized_axes = _normalize(joint_axes.to(device=joint_rot6d.device, dtype=joint_rot6d.dtype))
    axis_shape = (1,) * (joint_rot6d.ndim - 2) + normalized_axes.shape
    expanded_axes = normalized_axes.view(axis_shape)

    cos_theta = ((torch.diagonal(rotmat, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5).clamp(-1.0, 1.0)
    skew_vec = torch.stack(
        (
            rotmat[..., 2, 1] - rotmat[..., 1, 2],
            rotmat[..., 0, 2] - rotmat[..., 2, 0],
            rotmat[..., 1, 0] - rotmat[..., 0, 1],
        ),
        dim=-1,
    )
    sin_theta = 0.5 * (skew_vec * expanded_axes).sum(dim=-1)
    return torch.atan2(sin_theta, cos_theta)


def build_smp_feature_components(
    *,
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    root_lin_vel_w: torch.Tensor,
    root_ang_vel_w: torch.Tensor,
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    joint_axes: torch.Tensor,
    ee_pos_w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """构造一帧 SMP 所需的 heading-local 特征块。"""
    if root_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected root_pos_w last dim 3, got {root_pos_w.shape[-1]}")
    if root_quat_w.shape[-1] != 4:
        raise ValueError(f"Expected root_quat_w last dim 4, got {root_quat_w.shape[-1]}")
    if root_lin_vel_w.shape[-1] != 3 or root_ang_vel_w.shape[-1] != 3:
        raise ValueError("Expected root linear/angular velocities to have last dim 3")
    if ee_pos_w.shape[-1] != 3:
        raise ValueError(f"Expected ee_pos_w last dim 3, got {ee_pos_w.shape[-1]}")

    heading_rotation = build_heading_frame_rotation(root_quat_w)
    ee_pos_rel_w = ee_pos_w - root_pos_w.unsqueeze(-2)
    return {
        "base_lin_vel_b": world_to_local_frame(heading_rotation, root_lin_vel_w),
        "base_ang_vel_b": world_to_local_frame(heading_rotation, root_ang_vel_w),
        "joint_rot6d_rel": joint_positions_to_rot6d(joint_pos, default_joint_pos, joint_axes),
        "ee_pos_b": world_to_local_frame(heading_rotation, ee_pos_rel_w),
    }


def pack_smp_frame_features(
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    joint_rot6d_rel: torch.Tensor,
    ee_pos_b: torch.Tensor,
    expected_feature_dim: int | None = None,
) -> torch.Tensor:
    """将单帧 SMP 特征按照统一顺序拼接成向量。"""
    lead_shape = base_lin_vel_b.shape[:-1]
    if base_lin_vel_b.shape[-1] != 3:
        raise ValueError(f"Expected base_lin_vel_b last dim 3, got {base_lin_vel_b.shape[-1]}")
    if base_ang_vel_b.shape[-1] != 3:
        raise ValueError(f"Expected base_ang_vel_b last dim 3, got {base_ang_vel_b.shape[-1]}")
    if base_ang_vel_b.shape[:-1] != lead_shape:
        raise ValueError("SMP 根速度特征输入的前导维度不一致")
    if joint_rot6d_rel.shape[:-2] != lead_shape or ee_pos_b.shape[:-2] != lead_shape:
        raise ValueError("SMP 关节 rot6d 或末端执行器位置输入的前导维度不一致")
    if joint_rot6d_rel.shape[-1] != 6:
        raise ValueError(f"Expected joint_rot6d_rel last dim 6, got {joint_rot6d_rel.shape[-1]}")
    if ee_pos_b.shape[-1] != 3:
        raise ValueError(f"Expected ee_pos_b last dim 3, got {ee_pos_b.shape[-1]}")

    joint_rot6d_rel = joint_rot6d_rel.reshape(*lead_shape, -1)
    ee_pos_b = ee_pos_b.reshape(*lead_shape, -1)
    features = torch.cat([base_lin_vel_b, base_ang_vel_b, joint_rot6d_rel, ee_pos_b], dim=-1)
    if expected_feature_dim is not None and features.shape[-1] != expected_feature_dim:
        raise ValueError(f"Expected SMP feature dim {expected_feature_dim}, got {features.shape[-1]}")
    return features
