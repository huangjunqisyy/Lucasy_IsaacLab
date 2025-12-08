# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        # 1. 线速度追踪 (最重要，占 50%)
        self.rewards.track_lin_vel_xy.weight = 1.2  # 原 1.0
        # 2. 角速度追踪 (次重要，占 30%)
        self.rewards.track_ang_vel_z.weight = 0.5   # 原 0.8
        # 3. 辅助动作奖励 (占 20%)
        self.rewards.gait.weight = 0.35              # 原 0.3
        self.rewards.feet_clearance.weight = 0.1    # 原 0.4
        self.rewards.feet_air_time.weight = 0.5      # 原 0.25
        # ---------------- [移除的奖励] ----------------
        # 4. 存活奖励 (直接移除)
        # 强迫 Agent 只有“动起来”并且“动得对”才有分，站着不动没分
        self.rewards.alive.weight = 0.0             # 原 0.1
        # ---------------- [惩罚项 (按比例缩小)] ----------------
        # 既然正向奖励缩小了，惩罚项也要相应缩小，否则 Agent 会因为太怕扣分而不敢动
        # 稳定性惩罚
        self.rewards.flat_orientation_l2.weight = -5.0  # 原 -0.5 (防止乱晃)
        # self.rewards.base_height.weight = -8.0        # 原 -0.5 (防止蹲太低或跳太高)
        self.rewards.feet_slide.weight = -0.2           # 原 -0.5 (防止滑步，AMP 其实能自动学会这个)
        self.rewards.undesired_contacts.weight = -1.0   # 原 -0.5 (防止膝盖/手着地)
        # 能量与动作平滑惩罚 (保持微量)
        self.rewards.action_rate.weight = -0.05        # 原 -0.01
        self.rewards.joint_acc.weight = -2.5e-7         # 原 -0.01 (加速度通常数值很大，权重需要给很小)
        self.rewards.joint_vel.weight = -0.001             # 原 -0.1 (AMP 已经约束了速度，通常不需要额外惩罚)
        # 垂直速度惩罚 (防止跳跃)
        self.rewards.base_linear_velocity.weight = -2.0 # 原 -0.1 (惩罚 z 轴速度)
        self.rewards.base_angular_velocity.weight = -0.05 # 原 -0.1 (惩罚 xy 轴角速度)
        # 关节限制与偏差
        self.rewards.dof_pos_limits.weight = -5.0       # 原 -0.5
        self.rewards.joint_deviation_arms.weight = -0.1 # 原 -0.05
        self.rewards.joint_deviation_waists.weight = -1.0
        self.rewards.joint_deviation_legs.weight = -1.0

        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 0.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        # self.rewards.dof_torques_l2.weight = -2.0e-6
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


class G1FlatEnvCfg_PLAY(G1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
