# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import numpy as np

# def sigmoid(x, value_at_1):
#     scale = np.sqrt(-2 * np.log(value_at_1))
#     return torch.exp(-0.5 * (x*scale)**2)


# def tolerance(x, bounds=(0.0, 0.0), margin=0.0, value_at_margin=0.1):
#     lower, upper = bounds 
#     assert lower < upper
#     assert margin >= 0

#     in_bounds = torch.logical_and(lower <= x, x <= upper)
#     if margin == 0:
#         value = torch.where(in_bounds, 1.0, 0)
#     else:
#         d = torch.where(x < lower, lower - x, x - upper) / margin
#         value = torch.where(in_bounds, 1.0, sigmoid(d.double(), value_at_margin))
    
#     return value

def tolerance(x: torch.Tensor, bounds: tuple[float, float] = (0.0, 0.0), margin: float = 0.0, value_at_margin: float = 0.1) -> torch.Tensor:
    """
    计算基于高斯的 Tolerance 奖励。
    
    Args:
        x: 输入张量
        bounds: (lower, upper) 目标范围。在此范围内奖励为 1.0
        margin: 容差边距。超出 bounds 后，奖励随距离按高斯衰减
        value_at_margin: 当 x 刚好在 bounds ± margin 位置时的奖励值
    """
    lower, upper = bounds
    assert lower <= upper, "Lower bound must be <= upper bound"
    assert margin >= 0, "Margin must be non-negative"
    
    # 1. 判断是否在目标范围内
    in_bounds = (x >= lower) & (x <= upper)
    
    # 如果 margin 为 0，直接返回 0/1 奖励
    if margin == 0:
        return torch.where(in_bounds, torch.ones_like(x), torch.zeros_like(x))

    # 2. 计算距离 (利用 ReLU 简化逻辑，避免嵌套 where)
    # d = max(0, lower - x) + max(0, x - upper)
    d = torch.clamp(lower - x, min=0.0) + torch.clamp(x - upper, min=0.0)

    # 3. 计算高斯缩放系数
    # 我们希望: exp( -0.5 * (d_margin * scale)^2 ) = value_at_margin
    # 其中 d_margin = margin。
    # 解得 scale = sqrt( -2 * log(value) ) / margin
    # 因此最终指数项系数 k = -log(value) / margin^2
    # 使用 torch.tensor 避免重复创建 float 对象，并保证设备一致性
    
    value_at_margin = max(1e-6, value_at_margin) # 防止 log(0)
    
    # 这里的数学等价于: scale = sqrt(-2 * log(v)) / margin; value = exp(-0.5 * (d * scale)**2)
    # 简化后如下:
    factor = -np.log(value_at_margin) / (margin ** 2 + 1e-8)
    
    # 4. 计算奖励
    # value = exp( - d^2 * factor )
    value = torch.exp(- (d ** 2) * factor)
    
    # 5. 修正范围内为 1.0
    # 虽然 d=0 时 exp(0)=1，但在浮点运算中显式覆盖更安全
    value = torch.where(in_bounds, torch.ones_like(value), value)
    
    return value

"""
General.
"""


# 只要环境未终止，每一步都给予正向奖励。
def is_alive(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for being alive."""
    return (~env.termination_manager.terminated).float()


# 如果回合结束（且不是因为时间耗尽），给予惩罚。
def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


# 针对特定的终止原因（通过正则匹配 term_keys）进行惩罚。
class is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # find and store the termination terms
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        # Return the unweighted reward for the termination terms
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # Sums over terminations term values to account for multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""

# 抑制机器人在垂直方向上的不必要运动
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

# 惩罚机器人基座（Base）在 X 轴和 Y 轴上的角速度，即抑制机器人的“侧倾（Roll）”和“俯仰（Pitch）”晃动，但允许“转向（Yaw）”
def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


# 惩罚重力向量在机器人坐标系 XY 平面上的投影分量。强制机器人保持直立，防止身体倾斜。
def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

# 惩罚重力向量在机器人坐标系 XY 平面上的投影分量。强制机器人保持直立，防止身体倾斜。
def standup_flat_orientation_l2(env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    root_height = asset.data.root_pos_w[:, 2]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * (root_height >= target_height).float()

def reward_up_orientation(
    env: ManagerBasedRLEnv, 
    mode: str = "tolerance",  # "gaussian" (原代码的not gaussian分支) 或 "tolerance"
    sigma: float = 0.2,       # for "gaussian" mode
    threshold: float = 0.1,   # for "tolerance" mode (orientation_threshold)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Adapted HOST orientation reward for Isaac Lab.
    Encourages the robot to maintain an upright posture.
    """
    # 1. 获取资产数据
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 2. 获取状态
    # Isaac Lab 中 projected_gravity_b 是重力在基座系下的投影
    # 直立时，重力向下，在 Base 系中为 [0, 0, -1]
    pg = asset.data.projected_gravity_b 
    root_height = asset.data.root_pos_w[:, 2]

    grav_x = pg[:, 0]
    valid_face_orientation = (grav_x < 0.1).float()

    # 3. 计算奖励
    if mode == "gaussian": # 对应原代码的 if not self.is_gaussian (逻辑反转了名字，这里用 mode 区分更清晰)
        # 原逻辑：exp(mse / sigma) * (height > 0.4)
        # 修改目标为 [0, 0, -1] 以匹配 Isaac Lab 的物理定义
        target_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device)
        
        # 计算 MSE 误差
        mse_error = torch.sum(torch.square(pg - target_vec), dim=-1)
        
        # 计算奖励
        reward = torch.exp(-mse_error / sigma) # 注意：通常是 exp(-error), 原代码是 exp(error/sigma) 可能是笔误或 sigma 是负数？通常我们用 exp(-err/sigma)
        
        # 加上高度门控
        condition = (root_height > 0.4)
        reward = reward * condition.float()
        
    else: # mode == "tolerance" (HOST Paper Default)
        # HOST 论文逻辑：
        # tolerance(-projected_gravity_z, [threshold, inf], ...)
        # 直立时 pg_z 为 -1。 -pg_z 为 1。
        # 我们希望 -pg_z 接近 1 (即 pg_z 接近 -1)
        
        val = -pg[:, 2]
        
        # 仅在 Phase 1 (Righting) 高度以上生效？
        # 原代码：base_height > target_base_height_phase1，但这行代码在原 snippet 里似乎没用到？
        # HOST 论文 Table VI 显示 Base orientation 权重是 1，且有 f_tol。
        # 假设这里我们直接计算 tolerance reward

        reward = tolerance(
            val, 
            bounds=(1.0 - threshold, float('inf')), # 目标是接近 1
            margin=0.9, # 这里的 margin 需要根据具体调优，论文建议 func 参数
            value_at_margin=0.05
        )
        
    return reward * valid_face_orientation

def reward_face_up_orientation(
    env: ManagerBasedRLEnv, 
    threshold_height: float = 0.4, # 只有低于此高度（倒地时）才生效
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    鼓励机器人在倒地时保持'脸朝上' (Face Up / Supine) 的姿态，
    或者是惩罚'脸朝下' (Face Down / Prone) 的姿态。
    """
    # 1. 获取数据
    asset: RigidObject = env.scene[asset_cfg.name]
    pg = asset.data.projected_gravity_b 
    root_height = asset.data.root_pos_w[:, 2]

    # 2. 提取重力在 X 轴（前后方向）的分量
    # 如果 grav_x > 0: 重力指向前方 -> 脸朝下 (Face Down)
    # 如果 grav_x < 0: 重力指向后方 -> 脸朝上 (Face Up)
    grav_x = pg[:, 0]

    # 3. 计算奖励/惩罚
    # 逻辑 A (推荐): 强力惩罚脸朝下。
    # 我们希望 grav_x 越小越好。
    # 我们可以简单地惩罚 grav_x 为正的部分。
    # torch.clamp(grav_x, min=0.0) 会把脸朝上(负数)的情况变成0(无惩罚)，脸朝下(正数)变成正值(惩罚)。
    # 配合权重 weight = -1.0 使用。
    penalty = torch.clamp(grav_x, min=0.0)
    
    # 只有当机器人趴在地上时，我们才在乎它是仰卧还是俯卧。
    # 站起来后 (height > threshold)，grav_x 自然接近0，不需要干预。
    height_factor = torch.clamp(threshold_height - root_height, min=0.0)
    height_factor = height_factor / threshold_height  # 归一化到 [0, 1]

    # 返回惩罚项 (如果是逻辑A，记得在配置文件里 weight 设为负数)
    return penalty * height_factor

# 让机器人保持特定的行走高度（如蹲伏或站立）。
def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    # return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    # 只惩罚低于目标的情况
    return torch.square(torch.clamp(adjusted_target_height - asset.data.root_pos_w[:, 2], min=0.0))

def stand_height_reached(
    env: ManagerBasedRLEnv,
    target_height: float,
    threshold: float = 0.05,  # 允许的误差范围，比如目标0.5m，只要达到0.45m就算成功
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    当机器人相对于地面的高度达到目标高度时给予奖励 (Binary Reward)。
    用于判断机器人是否成功站立。

    Args:
        target_height: 期望的站立高度（相对于地面）。
        threshold: 判定成功的容差。例如 target=0.5, threshold=0.05, 则高度 > 0.45 即给奖励。
        asset_cfg: 机器人的配置项名称。
        sensor_cfg: 高度传感器（RayCaster）配置项名称。如果为None，则假设地面在 z=0。
    
    Returns:
        torch.Tensor: 如果达到高度返回 1.0，否则返回 0.0。
    """
    # 1. 获取机器人的资产
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos_z = asset.data.root_pos_w[:, 2]  # (num_envs,)

    # 2. 处理地形高度
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 获取射线击中点的 Z 轴平均值作为当前地面高度
        # sensor.data.ray_hits_w shape: (num_envs, num_rays, 3)
        ground_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # 平坦地形假设地面为 0
        ground_height = torch.zeros_like(root_pos_z)

    # 3. 计算相对高度
    current_rel_height = root_pos_z - ground_height

    # 4. 计算奖励
    # 逻辑：如果 当前高度 >= (目标高度 - 容差)，则认为站起来了
    # 使用 float() 将布尔值转换为 0.0 或 1.0
    is_standing = (current_rel_height >= (target_height - threshold)).float()

    return is_standing

def stand_height_reward(
    env: ManagerBasedRLEnv,
    mode: str = "tolerance",
    target_height: float = 0.65,
    threshold: float = 0.05,  
    target_margin: float = 0.5,
    gaussian_scale: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """
    基于相对高度的站立奖励 (Isaac Lab 版本)。
    """
    # 1. 获取机器人资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    root_pos_z = asset.data.root_pos_w[:, 2]  # (num_envs,)

    # 2. 处理地形高度
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # 获取射线击中点的 Z 轴平均值作为当前地面高度
        # sensor.data.ray_hits_w shape: (num_envs, num_rays, 3)
        ground_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # 平坦地形假设地面为 0
        ground_height = torch.zeros_like(root_pos_z)

    # 3. 计算相对高度
    current_rel_height = root_pos_z - ground_height

    # 3. 分支逻辑
    if mode == "binary":
        # 对应原代码: if not self.is_gaussian
        # 原逻辑是对 height 进行 clamp(0, 1)
        # 这里假设直接返回高度作为奖励（归一化可能需要在外部处理，或者这里硬编码）
        return (current_rel_height >= (target_height - threshold)).float()
    
    else:
        # 6. 计算高斯奖励 (Tolerance)
        # 原代码逻辑: reward = tolerance(head_height, (target, inf), margin, 0.1)
        # 意味着: 只要 > target, reward = 1.0; 否则按高斯衰减
        val = current_rel_height
        reward = tolerance(
            val, 
            bounds=(target_height - threshold, float('inf')), 
            margin=target_margin, 
            value_at_margin=gaussian_scale
        )
        
        return reward

# 惩罚特定身体部件的线性加速度。防止动作过猛，减少机械冲击，使运动更平滑。
def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :], dim=-1), dim=1)


"""
Joint penalties.
"""


# 最小化力矩。节能。鼓励机器人用最小的力完成任务。
def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


# 惩罚关节速度的绝对值和。抑制高频震荡，鼓励低速运动
def joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


# 惩罚关节运动过快或高频震荡
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


# 限制关节加速度。极大地提高动作的平滑性，减少急停急起。
def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


# 维持默认姿态。惩罚当前关节角度偏离默认（标称）姿态的程度。让机器人在无指令时倾向于回到“舒适”或“自然”的姿势。
def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


# 关节软限位。
def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


# 速度软限位。防止电机超速。
def joint_vel_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.

    Args:
        soft_ratio: The ratio of the soft limits to be used.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
        - asset.data.soft_joint_vel_limits[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to max error = 1 rad/s per joint to avoid huge penalties
    out_of_limits = out_of_limits.clip_(min=0.0, max=1.0)
    return torch.sum(out_of_limits, dim=1)

# 当速度指令接近 0 时，才惩罚偏离默认姿态的行为
def stand_still_soft(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    当指令速度为 0 时，惩罚关节偏离默认位置的行为。
    """
    # 1. 获取当前的指令 (假设 command_name 是 "base_velocity")
    # shape: (num_envs, 3) -> [v_x, v_y, w_z]
    commands = env.command_manager.get_command("base_velocity")
    
    # 2. 计算指令的模长 (速度大小)
    # 这里我们只关心线速度和角速度的总和
    cmd_norm = torch.norm(commands, dim=1)
    
    # 3. 制作一个“软开关” (Soft Switch)
    # 当 cmd_norm 接近 0 时，is_idle 接近 1.0
    # 当 cmd_norm 变大时，is_idle 迅速衰减到 0.0
    # sigma 控制带宽，0.1 意味着只要有一点点速度指令，这个惩罚就立刻消失
    is_idle = torch.exp(-cmd_norm.square() / 0.01)
    
    # 4. 计算关节偏差 (复制你之前的 joint_deviation_l1 逻辑)
    asset = env.scene[asset_cfg.name]
    # 计算当前角度与默认角度的 L1 距离
    deviation = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    
    # 5. 返回结果：只有在 idle 状态下，才返回偏差值
    # 注意：这里返回正数代表偏差量，我们在 Config 里给负权重把它变成惩罚
    return is_idle * deviation

def stand_still(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < 0.1)


"""
Action penalties.
"""

# 力矩超限惩罚。惩罚实际施加力矩与计算力矩的差异（主要针对显式执行器）。确保指令在物理上是可执行的（Sim-to-Real 重要项）。
def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


# 动作变化率惩罚。惩罚当前动作与上一帧动作的差值平方。防止控制信号高频抖动，保护电机。
def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


# 动作幅度惩罚。鼓励输出较小的控制信号，避免饱和。
def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


"""
Contact sensor.
"""


# 非预期碰撞。防止机器人摔倒、磕碰膝盖或自行碰撞（Self-collision）。
def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1)

# 站立后非预期碰撞。防止机器人摔倒、磕碰膝盖或自行碰撞（Self-collision）。
def standup_undesired_contacts(env: ManagerBasedRLEnv, threshold: float, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None,) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold

    asset: RigidObject = env.scene[asset_cfg.name]
    
    root_height = asset.data.root_pos_w[:, 2]  # (num_envs,)
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1) * (root_height >= target_height).float()

# 持续的非预期接触。惩罚机器人身体部件长时间接触地面，防止机器人“赖”在地上。
def long_time_undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, tolerance_ratio: float = 0.2) -> torch.Tensor:
    """
    Penalize non-foot contacts based on the duration/frequency of contact in the history buffer.
    Helps the robot learn to stand up by penalizing 'lazy' body contacts over time.
    """
    # 1. 获取传感器数据
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. 获取带历史记录的接触力数据
    # Shape: (num_envs, history_length, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    # 3. 计算接触力的模长 (Magnitude)
    # Shape: (num_envs, history_length, num_bodies)
    forces_norm = torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1)
    
    # 4. 判断每一帧是否发生接触 (0 或 1)
    # Shape: (num_envs, history_length, num_bodies)
    is_contacting = (forces_norm > threshold).float()
    
    # 5. 计算接触的“持续度” (平均值)
    # 在时间维度 (dim=1) 上取平均。
    # 结果含义：在过去的历史窗口内，该部位有百分之多少的时间是在接触地面的。
    # 0.0 表示完全悬空，1.0 表示一直贴地，0.1 表示只有短时间接触（如借力瞬间）
    contact_duration_ratio = torch.mean(is_contacting, dim=1)

    # 6. 减去容忍度，并过滤负数
    # 如果比例是 0.15 (少于 0.2)，结果为 0 (不惩罚)
    # 如果比例是 0.50 (多于 0.2)，结果为 0.3 (惩罚超出的部分)
    excess_contact = torch.relu(contact_duration_ratio - tolerance_ratio)
    # ===========================================

    # 7. 求和
    return torch.sum(excess_contact, dim=1)


# 接触奖励/约束。强迫机器人脚必须着地（较少用，通常用于特定阶段）。
def desired_contacts(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    """Penalize if none of the desired contacts are present."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    zero_contact = (~contacts).all(dim=1)
    return 1.0 * zero_contact


# 接触力过大。防止机器人跺脚过重，保护力传感器或地面。
def contact_forces(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces as the amount of violations of the net contact force."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # compute the violation
    violation = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] - threshold
    # compute the penalty
    return torch.sum(violation.clip(min=0.0), dim=1)


"""
Velocity-tracking rewards.
"""


# 线速度追踪。奖励机器人实际 XY 速度接近指令速度的程度。
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


# 角速度追踪。奖励机器人实际 Z 轴角速度（转向）接近指令速度的程度。
def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def shank_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    knee_regex: str = ".*knee_link",  # 正则表达式
    foot_regex: str = ".*ankle_roll.*",  # 正则表达式
    target_height_phase1: float = 0.45, # 只有当基座高于此高度才开始计算对齐奖励
    target_height_phase3: float = 0.65, # 任务完成的高度
    margin: float = 1.0,
    value_at_margin: float = 0.1,
    use_post_task_override: bool = True,
) -> torch.Tensor:
    """
    奖励小腿（Shank）垂直于地面。
    """
    # 1. 获取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 2. 解析刚体索引 (使用 find_bodies)
    # 这会返回符合正则的所有刚体索引
    knee_indices, _ = asset.find_bodies(knee_regex)
    foot_indices, _ = asset.find_bodies(foot_regex)
    
    # 3. 获取位置数据 (num_envs, num_bodies, 3)
    knee_pos = asset.data.body_pos_w[:, knee_indices, :3]
    foot_pos = asset.data.body_pos_w[:, foot_indices, :3]
    
    # 4. 计算向量与垂直度
    # 假设 knee 和 foot 的数量是一一对应的
    vec = knee_pos - foot_pos
    vec_norm = torch.norm(vec, dim=-1, keepdim=True)
    vec_norm = torch.where(vec_norm < 1e-6, torch.ones_like(vec_norm), vec_norm) # 避免除0
    
    # 取 Z 分量 (Unit vector Z component)
    shank_z = (vec[:, :, 2:3] / vec_norm).squeeze(-1) # shape: (num_envs, num_legs)

    # 5. 计算所有腿的平均方向
    feet_orientation = torch.mean(shank_z, dim=-1) # shape: (num_envs,)

    # 6. 计算奖励 (Tolerance)
    # 目标是接近 1.0 (垂直)，下限设为 0.8
    reward = tolerance(
        feet_orientation, 
        bounds=(0.8, float('inf')), 
        margin=margin, 
        value_at_margin=value_at_margin
    )
    
    # 7. 应用 Phase 1 高度门控 (只有稍微站起来一点才开始算这个奖励)
    root_height = asset.data.root_pos_w[:, 2]
    base_height_condition = (root_height > target_height_phase1).float()
    reward = reward * base_height_condition

    # 8. Post Task Override (Phase 3)
    # 如果完全站直了，给予满分奖励，防止抖动导致的奖励下降
    if use_post_task_override:
        standup = root_height > target_height_phase3
        reward = torch.where(standup, torch.ones_like(reward), reward)

    return reward 

def ground_parallel_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    # 需要区分左脚和右脚，以实现“局部平整度”检测
    left_ankle_regex: str = ".*left_ankle_roll.*", 
    right_ankle_regex: str = ".*right_ankle_roll.*",
    target_height_phase3: float = 0.5,
    var_threshold: float = 0.05,
    use_post_task_override: bool = True,
) -> torch.Tensor:
    """
    奖励双脚平整度（分别计算左右脚方差，允许双脚高度不同，但要求各自平行于地面）。
    """
    # 1. 获取资产
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # 2. 分别获取左右脚索引
    # 注意：为了让方差计算有效，每个 regex 最好匹配到 >1 个刚体（如 toes + heel），
    # 或者如果只是单个刚体，这种基于位置方差的方法其实是无效的（方差为NaN或0）。
    # 原代码能工作暗示了 indices 里有多个点。
    left_indices, _ = asset.find_bodies(left_ankle_regex)
    right_indices, _ = asset.find_bodies(right_ankle_regex)
    
    # 3. 获取 Z 坐标并放大
    # (num_envs, num_points_left)
    left_z = asset.data.body_pos_w[:, left_indices, 2] * 10.0
    # (num_envs, num_points_right)
    right_z = asset.data.body_pos_w[:, right_indices, 2] * 10.0
    
    # 4. 分别计算方差
    # 注意：如果某个列表里只有 1 个 body，var 会返回 NaN。需要处理这种情况。
    # 这里假设 regex 能匹配到多个点，或者原逻辑本身依赖这一假设。
    if len(left_indices) > 1:
        var_left = torch.var(left_z, dim=1)
    else:
        var_left = torch.zeros(env.num_envs, device=env.device) # 单点无法计算平面度，默认为平
        
    if len(right_indices) > 1:
        var_right = torch.var(right_z, dim=1)
    else:
        var_right = torch.zeros(env.num_envs, device=env.device)

    # 5. 合并方差 (取平均，或者像原代码那样 concat 后取 mean)
    # 原代码: mean(concat([left_var, right_var]))
    combined_var = (var_left + var_right) / 2.0
    
    # 6. 计算奖励
    reward = (combined_var < var_threshold).float()

    # 7. Post Task Override
    if use_post_task_override:
        root_height = asset.data.root_pos_w[:, 2]
        standup = root_height > target_height_phase3
        reward = torch.where(standup, torch.ones_like(reward), reward)
        
    return reward

def foot_flat_orientation_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_regex: str = ".*ankle_roll.*", # 匹配双脚
    target_height_phase3: float = 0.5,
    use_post_task_override: bool = True,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_indices, _ = asset.find_bodies(foot_regex)
    
    # 获取脚的四元数 (num_envs, num_feet, 4)
    foot_quat = asset.data.body_quat_w[:, foot_indices, :]

    # 获取脚的数量
    num_feet = len(foot_indices)
    
    # 计算重力在脚部坐标系的投影
    # 如果脚是平的，重力应该垂直于脚面（即投影接近 [0, 0, -1] 或 [0, 0, 1]，取决于建模轴向）
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    gravity_vec = gravity_vec.expand(env.num_envs, len(foot_indices), 3).contiguous() # (N, num_feet, 3)
    projected_gravity = quat_apply(foot_quat, gravity_vec) # (N, num_feet, 3)
    
    # 假设脚的 Z 轴应该垂直地面：
    # 我们希望 projected_gravity[:, :, 2] 接近 1.0 (或 -1.0)
    flatness_error = torch.sum(torch.square(projected_gravity[..., 2] + 1.0), dim=1) # 简单的 MSE
    
    reward = torch.exp(-flatness_error * 5.0) # 转化为 [0, 1] 奖励
    
    if use_post_task_override:
        root_height = asset.data.root_pos_w[:, 2]
        standup = root_height > target_height_phase3
        reward = torch.where(standup, torch.ones_like(reward), reward)

    return reward