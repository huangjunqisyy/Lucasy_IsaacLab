# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rsl_rl.diffusion.gsi import SMPResetReference, SMPResetState

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _normalize_env_ids(env_ids, total_envs: int, device: torch.device) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(total_envs, device=device, dtype=torch.long)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def build_smp_reset_reference(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
    asset_name: str = "robot",
) -> SMPResetReference:
    """从环境默认状态提取 GSI 解码所需的参考 root/joint 状态。"""
    target_env = _unwrap_env(env)
    asset = target_env.scene[asset_name]
    device = asset.data.default_root_state.device
    env_ids = _normalize_env_ids(env_ids, asset.data.default_root_state.shape[0], device=device)
    default_root_state = asset.data.default_root_state[env_ids]
    return SMPResetReference(
        root_pos_w=default_root_state[:, :3].clone(),
        root_quat_w=default_root_state[:, 3:7].clone(),
        joint_pos=asset.data.default_joint_pos[env_ids].clone(),
        joint_vel=asset.data.default_joint_vel[env_ids].clone(),
    )


def apply_smp_reset_state(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    state: SMPResetState,
    asset_name: str = "robot",
) -> None:
    """把 GSI 生成的 reset state 写回仿真，并同步位置/速度目标。"""
    target_env = _unwrap_env(env)
    asset = target_env.scene[asset_name]
    device = state.root_pos_w.device
    env_ids = _normalize_env_ids(env_ids, state.root_pos_w.shape[0], device=device)

    root_pose = torch.cat((state.root_pos_w, state.root_quat_w), dim=-1)
    root_velocity = torch.cat((state.root_lin_vel_w, state.root_ang_vel_w), dim=-1)
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
    asset.write_joint_state_to_sim(state.joint_pos, state.joint_vel, env_ids=env_ids)
    # 同步 PD/velocity target，避免 reset 后第一步仍使用旧目标。
    if hasattr(asset, "set_joint_position_target"):
        asset.set_joint_position_target(state.joint_pos, env_ids=env_ids)
    if hasattr(asset, "set_joint_velocity_target"):
        asset.set_joint_velocity_target(state.joint_vel, env_ids=env_ids)
