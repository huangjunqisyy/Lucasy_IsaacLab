# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .smp_features import build_smp_feature_components, pack_smp_frame_features

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def smp_frame_features(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    key_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_axes: torch.Tensor | list[list[float]] | list[tuple[float, float, float]] | None = None,
    expected_joint_dim: int | None = None,
    expected_feature_dim: int | None = None,
) -> torch.Tensor:
    """在 pelvis-heading 局部坐标系下提取一帧 SMP 特征。"""
    if asset_cfg.name != ee_asset_cfg.name or asset_cfg.name != key_body_cfg.name:
        raise ValueError("SMP 观测当前要求 asset_cfg、ee_asset_cfg 和 key_body_cfg 指向同一个机器人资产")
    if joint_axes is None:
        raise ValueError("SMP 观测需要显式提供 joint_axes 以编码 joint rot6d")
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if expected_joint_dim is not None and joint_pos.shape[-1] != expected_joint_dim:
        raise ValueError(f"Expected SMP joint dim {expected_joint_dim}, got {joint_pos.shape[-1]}")

    joint_axes_tensor = torch.as_tensor(joint_axes, dtype=joint_pos.dtype, device=joint_pos.device)
    feature_components = build_smp_feature_components(
        root_pos_w=asset.data.root_pos_w,
        root_quat_w=asset.data.root_quat_w,
        root_lin_vel_w=asset.data.root_lin_vel_w,
        root_ang_vel_w=asset.data.root_ang_vel_w,
        joint_pos=joint_pos,
        default_joint_pos=default_joint_pos,
        joint_axes=joint_axes_tensor,
        ee_pos_w=asset.data.body_pos_w[:, ee_asset_cfg.body_ids],
    )

    return pack_smp_frame_features(
        base_lin_vel_b=feature_components["base_lin_vel_b"],
        base_ang_vel_b=feature_components["base_ang_vel_b"],
        joint_rot6d_rel=feature_components["joint_rot6d_rel"],
        ee_pos_b=feature_components["ee_pos_b"],
        expected_feature_dim=expected_feature_dim,
    )
