# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# 腾空时间奖励。奖励机器人迈出“大步”。
def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# 奖励的前提是只有一只脚着地。鼓励双足机器人通过“单腿支撑相”进行行走，而不是双脚跳跃或双脚拖地，促进周期性的左右交替步态。
def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


# 滑步惩罚
def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


# XY平面线速度追踪。忽略了机器人的侧倾（Roll）和俯仰（Pitch）对速度测量的干扰，只关注“机器人朝向的前方”和“侧方”的速度是否达标。
def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


# 确保机器人能够按照指令准确地左转或右转
def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


# 当用户松开摇杆让机器人停下时，机器人应该回到自然、美观的站立姿态
def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


# 鼓励机器人在迈腿（摆动）过程中将脚抬高到指定高度。
# 核心思路：只在脚正在水平移动（摆动相）时，才惩罚脚离目标高度的偏差；
# 脚静止（支撑相）时不关心高度，从而只对"迈步抬腿"阶段生效。
#
# 参数说明：
#   target_height : 期望的脚离地高度（米），例如 0.05
#   std           : 高斯核的标准差，控制奖励对误差的敏感度（越小越严格）
#   tanh_mult     : tanh 的缩放系数，控制"速度门控"的灵敏度
#
# 计算步骤：
#   1. foot_z_target_error = (foot_z - target_height)²
#      每只脚当前 z 坐标与目标高度的平方误差（越偏离目标，值越大）
#   2. foot_velocity_tanh = tanh(tanh_mult * ‖v_xy‖)
#      脚在水平面的速度经 tanh 映射到 (0,1)，起"门控"作用：
#        - 脚静止时 ≈ 0 → 不产生惩罚（支撑相不管高度）
#        - 脚快速移动时 ≈ 1 → 全额计入高度误差（摆动相必须抬脚）
#   3. reward = exp( -Σ(error * gate) / std² )
#      对所有脚的加权误差求和，再用高斯核转为 (0,1] 的奖励：
#        - 所有摆动脚都在目标高度附近 → 奖励接近 1
#        - 偏差越大 → 奖励指数下降趋近 0
def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)

"""
Feet Gait rewards.
"""

# 强制步态相位奖励
def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,          # 一个完整步态周期的时长（秒），例如 0.5s
    offset: list[float],    # 每条腿的相位偏移量（0~1），例如 [0.0, 0.5] 表示两条腿交替（相差半个周期）
    sensor_cfg: SceneEntityCfg,  # 接触传感器配置，body_ids 指定哪些脚
    threshold: float = 0.5,      # 站立相占整个周期的比例，默认 0.5 表示站立和摆动各占一半
    command_name=None,           # 可选，若提供则在零指令时不给奖励
) -> torch.Tensor:
    # 1. 获取每只脚当前是否处于接触状态（True=触地）
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    # 2. 计算全局相位：用当前仿真时间对步态周期取模，归一化到 [0, 1)
    #    episode_length_buf 是当前 episode 已走的步数，乘以 step_dt 得到时间（秒）
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)

    # 3. 为每条腿加上各自的相位偏移，得到该腿当前在步态周期中的位置
    #    例如 offset=[0.0, 0.5]，左腿相位=global_phase，右腿相位=global_phase+0.5
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)  # shape: (num_envs, num_legs)

    # 4. 逐腿计算奖励：
    #    - 如果该腿的相位 < threshold → 期望处于"站立相"（应该触地）
    #    - 如果该腿的相位 >= threshold → 期望处于"摆动相"（应该腾空）
    #    - 用 XNOR（同或）判断：实际状态与期望状态一致时 +1
    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold   # 期望该腿此刻应触地？
        reward += ~(is_stance ^ is_contact[:, i]) # XNOR: 期望与实际一致 → True(1)，不一致 → False(0)

    # 5. 如果指定了 command_name，在零指令（机器人应原地站立）时将奖励清零
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward

def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)