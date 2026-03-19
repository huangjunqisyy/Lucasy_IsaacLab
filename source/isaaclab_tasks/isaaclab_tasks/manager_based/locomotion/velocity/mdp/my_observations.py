# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

from .smp_features import pack_smp_frame_features

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def smp_frame_features(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    key_body_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute SMP per-frame features in the robot root frame."""
    asset: Articulation = env.scene[asset_cfg.name]

    base_lin_vel_b = asset.data.root_lin_vel_b
    base_ang_vel_b = asset.data.root_ang_vel_b
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    ee_pos_w = asset.data.body_pos_w[:, ee_asset_cfg.body_ids]
    ee_quat_w = asset.data.body_quat_w[:, ee_asset_cfg.body_ids]
    root_pos_w = asset.data.root_pos_w.unsqueeze(1)
    root_quat_w = asset.data.root_quat_w.unsqueeze(1)
    root_quat_expanded = root_quat_w.expand_as(ee_quat_w)
    ee_pos_b = quat_apply_inverse(root_quat_expanded, ee_pos_w - root_pos_w)

    key_body_quat_w = asset.data.body_quat_w[:, key_body_cfg.body_ids]
    root_quat_inv = quat_conjugate(root_quat_w)
    root_quat_inv_expanded = root_quat_inv.expand_as(key_body_quat_w)
    key_body_quat_b = quat_mul(root_quat_inv_expanded, key_body_quat_w)

    return pack_smp_frame_features(
        base_lin_vel_b=base_lin_vel_b,
        base_ang_vel_b=base_ang_vel_b,
        joint_pos_rel=joint_pos_rel,
        ee_pos_b=ee_pos_b,
        key_body_quat_b=key_body_quat_b,
    )
