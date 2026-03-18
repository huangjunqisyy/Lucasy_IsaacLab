# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import gymnasium as gym
import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version

from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer

from .common import VecEnvStepReturn
from .manager_based_env import ManagerBasedEnv
from .manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

from isaaclab.utils.math import quat_apply

from isaaclab.sim.spawners.shapes import ConeCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import PreviewSurfaceCfg


class ManagerBasedRLEnv(ManagerBasedEnv, gym.Env):
    """The superclass for the manager-based workflow reinforcement learning-based environments.

    This class inherits from :class:`ManagerBasedEnv` and implements the core functionality for
    reinforcement learning-based environments. It is designed to be used with any RL
    library. The class is designed to be used with vectorized environments, i.e., the
    environment is expected to be run in parallel with multiple sub-environments. The
    number of sub-environments is specified using the ``num_envs``.

    Each observation from the environment is a batch of observations for each sub-
    environments. The method :meth:`step` is also expected to receive a batch of actions
    for each sub-environment.

    While the environment itself is implemented as a vectorized environment, we do not
    inherit from :class:`gym.vector.VectorEnv`. This is mainly because the class adds
    various methods (for wait and asynchronous updates) which are not required.
    Additionally, each RL library typically has its own definition for a vectorized
    environment. Thus, to reduce complexity, we directly use the :class:`gym.Env` over
    here and leave it up to library-defined wrappers to take care of wrapping this
    environment for their agents.

    Note:
        For vectorized environments, it is recommended to **only** call the :meth:`reset`
        method once before the first call to :meth:`step`, i.e. after the environment is created.
        After that, the :meth:`step` function handles the reset of terminated sub-environments.
        This is because the simulator does not support resetting individual sub-environments
        in a vectorized environment.

    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize the environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None, which
                is similar to ``"human"``.
        """
        # -- counter for curriculum
        self.common_step_counter = 0

        # initialize the episode length buffer BEFORE loading the managers to use it in mdp functions.
        self.episode_length_buf = torch.zeros(cfg.scene.num_envs, device=cfg.sim.device, dtype=torch.long)

        # initialize the base class to setup the scene.
        super().__init__(cfg=cfg)
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

        # 预先判断是否开启了辅助力课程
        if self.cfg.getup_curriculum:
            print("[INFO]: Get-Up Curriculum is ENABLED.")
            # === 初始化独立课程 Buffer ===
            # 1. 辅助力 Buffer (num_envs,)
            self.assist_force_buf = torch.full(
                (self.num_envs,), 
                self.cfg.initial_assist_force, 
                device=self.device
            )
            
            # 2. 动作缩放 Buffer (num_envs, 1) -> 注意维度，方便后续广播乘法
            self.action_scale_buf = torch.full(
                (self.num_envs, 1), 
                self.cfg.initial_action_scale, 
                device=self.device
            )

            robot = self.scene["robot"]
            target_body_name = "pelvis" # 你可以把它做成变量或从 cfg 读取

            try:
                # 查找 index
                self.idx = robot.body_names.index(target_body_name)
            except ValueError:
                print(f"[Warning] Body '{target_body_name}' not found. Defaulting to index 0.")
                self.idx = 0

            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/AssistForce",
                markers={
                    "arrow": ConeCfg(
                        radius=0.05, 
                        height=0.3, 
                        visual_material=PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)), # Red color
                    )
                }
            )
            self.assist_force_marker = VisualizationMarkers(marker_cfg)

    """
    Properties.
    """

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # call the parent class to load the managers for observations and actions.
        super().load_managers()

        # prepare the managers
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def setup_manager_visualizers(self):
        """Creates live visualizers for manager terms."""

        self.manager_visualizers = {
            "action_manager": ManagerLiveVisualizer(manager=self.action_manager),
            "observation_manager": ManagerLiveVisualizer(manager=self.observation_manager),
            "command_manager": ManagerLiveVisualizer(manager=self.command_manager),
            "termination_manager": ManagerLiveVisualizer(manager=self.termination_manager),
            "reward_manager": ManagerLiveVisualizer(manager=self.reward_manager),
            "curriculum_manager": ManagerLiveVisualizer(manager=self.curriculum_manager),
        }

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        if hasattr(self, "action_scale_buf"):
            # 确保 action 在正确的 device 上进行计算
            action = action.to(self.device)
            # 执行逐环境的缩放
            action = action * self.action_scale_buf

        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        # 核心循环：执行多次物理步
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            # 1. 应用动作
            self.action_manager.apply_action()

            if self.cfg.getup_curriculum:
                self._apply_assistive_force(self.idx)

            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            # 2. 物理步进
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            # 3. 按需渲染
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            # 4. 更新场景缓冲区 (位置、速度等)
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        # 如果有环境需要重置，计算当前的观测值（这代表了 termination 时的状态）
        # 注意：这里我们调用 compute 但通常不更新历史 buffer (update_history=False)，
        # 因为历史 buffer 的更新通常留给重置后的观测计算。
        if len(reset_env_ids) > 0:
            # 获取当前物理状态下的观测（即 s_T）
            terminal_obs = self.observation_manager.compute()
            
            # 将其保存到 extras 中，供算法层使用
            if self.extras is None: self.extras = {}
            # 注意：Isaac Lab 的 extras 通常每步会被清空或覆盖，确保这里是安全的写入
            self.extras["terminal_observation"] = {}
            if isinstance(terminal_obs, dict):
                for key, value in terminal_obs.items():
                    if value is not None:
                      self.extras["terminal_observation"][key] = value[reset_env_ids].to(self.device)
            else:
                self.extras["terminal_observation"] = terminal_obs[reset_env_ids].to(self.device)
        else:
            # 清理旧数据防止误用
            if self.extras and "terminal_observation" in self.extras:
                self.extras.pop("terminal_observation")

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command(更新指令，如随机生成新的目标速度)
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # 注意：这必须在 reset 之后做！
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        """Run rendering without stepping through the physics.

        By convention, if mode is:

        - **human**: Render to the current display and return nothing. Usually for human consumption.
        - **rgb_array**: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an
          x-by-y pixel image, suitable for turning into a video.

        Args:
            recompute: Whether to force a render even if the simulator has already rendered the scene.
                Defaults to False.

        Returns:
            The rendered image as a numpy array if mode is "rgb_array". Otherwise, returns None.

        Raises:
            RuntimeError: If mode is set to "rgb_data" and simulation render mode does not support it.
                In this case, the simulation render mode must be set to ``RenderMode.PARTIAL_RENDERING``
                or ``RenderMode.FULL_RENDERING``.
            NotImplementedError: If an unsupported rendering mode is specified.
        """
        # run a rendering step of the simulator
        # if we have rtx sensors, we do not need to render again sin
        if not self.sim.has_rtx_sensors() and not recompute:
            self.sim.render()
        # decide the rendering mode
        if self.render_mode == "human" or self.render_mode is None:
            return None
        elif self.render_mode == "rgb_array":
            # check that if any render could have happened
            if self.sim.render_mode.value < self.sim.RenderMode.PARTIAL_RENDERING.value:
                raise RuntimeError(
                    f"Cannot render '{self.render_mode}' when the simulation render mode is"
                    f" '{self.sim.render_mode.name}'. Please set the simulation render mode to:"
                    f"'{self.sim.RenderMode.PARTIAL_RENDERING.name}' or '{self.sim.RenderMode.FULL_RENDERING.name}'."
                    " If running headless, make sure --enable_cameras is set."
                )
            # create the annotator if it does not exist
            if not hasattr(self, "_rgb_annotator"):
                import omni.replicator.core as rep

                # create render product
                self._render_product = rep.create.render_product(
                    self.cfg.viewer.cam_prim_path, self.cfg.viewer.resolution
                )
                # create rgb annotator -- used to read data from the render product
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                self._rgb_annotator.attach([self._render_product])
            # obtain the rgb data
            rgb_data = self._rgb_annotator.get_data()
            # convert to numpy array
            rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data.size == 0:
                return np.zeros((self.cfg.viewer.resolution[1], self.cfg.viewer.resolution[0], 3), dtype=np.uint8)
            else:
                return rgb_data[:, :, :3]
        else:
            raise NotImplementedError(
                f"Render mode '{self.render_mode}' is not supported. Please use: {self.metadata['render_modes']}."
            )

    def close(self):
        if not self._is_closed:
            # destructor is order-sensitive
            del self.command_manager
            del self.reward_manager
            del self.termination_manager
            del self.curriculum_manager
            # call the parent class to close the environment
            super().close()

    """
    Helper functions.
    """

    def _configure_gym_env_spaces(self):
        """Configure the action and observation spaces for the Gym environment."""
        # observation space (unbounded since we don't impose any limits)
        self.single_observation_space = gym.spaces.Dict()
        for group_name, group_term_names in self.observation_manager.active_terms.items():
            # extract quantities about the group
            has_concatenated_obs = self.observation_manager.group_obs_concatenate[group_name]
            group_dim = self.observation_manager.group_obs_dim[group_name]
            # check if group is concatenated or not
            # if not concatenated, then we need to add each term separately as a dictionary
            if has_concatenated_obs:
                self.single_observation_space[group_name] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=group_dim)
            else:
                group_term_cfgs = self.observation_manager._group_obs_term_cfgs[group_name]
                term_dict = {}
                for term_name, term_dim, term_cfg in zip(group_term_names, group_dim, group_term_cfgs):
                    low = -np.inf if term_cfg.clip is None else term_cfg.clip[0]
                    high = np.inf if term_cfg.clip is None else term_cfg.clip[1]
                    term_dict[term_name] = gym.spaces.Box(low=low, high=high, shape=term_dim)
                self.single_observation_space[group_name] = gym.spaces.Dict(term_dict)
        # action space (unbounded since we don't impose any limits)
        action_dim = sum(self.action_manager.action_term_dim)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,))

        # batch the spaces for vectorized environments
        self.observation_space = gym.vector.utils.batch_space(self.single_observation_space, self.num_envs)
        self.action_space = gym.vector.utils.batch_space(self.single_action_space, self.num_envs)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        if len(env_ids) > 0 and self.cfg.getup_curriculum:
            self._update_curriculum(env_ids)
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            env_step_count = self._sim_step_counter // self.cfg.decimation
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=env_step_count)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- recorder manager
        info = self.recorder_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0

    def _apply_assistive_force(self, root_body_idx: int = 0):
        """
        当机器人接近直立时，施加向上的辅助拉力。
        """
        # 1. 获取机器人的姿态
        # 假设你的机器人是 scene 中的 "robot"
        robot = self.scene["robot"]
        root_quat = robot.data.root_quat_w  # (num_envs, 4)
        root_pos = robot.data.root_pos_w  # (num_envs, 3)
        root_vel = robot.data.root_lin_vel_w  # (num_envs, 3)
        root_ang_vel = robot.data.root_ang_vel_w
        
        # 2. 计算基座 Z 轴在世界坐标系下的方向
        # 机器人的局部 Z 轴向量 (0, 0, 1)
        # 将其旋转到世界坐标系
        params_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        heading_vec = quat_apply(root_quat, params_vec) # (num_envs, 3)
        
        # 3. 判断是否接近直立
        # heading_vec[:, 2] 是 Z 分量，接近 1 表示直立，接近 0 表示躺平
        # 这里的阈值 self.cfg.upright_threshold 比如设为 0.5 (45度) 或 0.7
        is_upright = heading_vec[:, 2] > self.cfg.upright_threshold
        target_stand_height = self.cfg.target_stand_height
        is_below_target_height = root_pos[:, 2] < target_stand_height
        should_apply_lift = torch.logical_and(is_upright, is_below_target_height)
        active_mask = should_apply_lift & (self.assist_force_buf > 1e-3)

        # damping_gain = 100.0 # 阻尼系数，越大“空气”越粘稠
        # damping_force = -root_vel * damping_gain
        # # 我们不希望阻尼影响 Z 轴的起立，只限制 XY 水平乱晃
        # damping_force[:, 2] = 0.0 
        
        # # 角速度阻尼 (Angular Damping) - 核心改动！
        # # 防止机器人像陀螺一样乱转
        # ang_damping_gain = 10.0 # 试着给 5.0 ~ 20.0
        # damping_torque = -root_ang_vel * ang_damping_gain

        forces_lift = torch.zeros((self.num_envs, 3), device=self.device)
        forces_lift[:, 2] = torch.where(
            active_mask, 
            self.assist_force_buf, 
            torch.zeros_like(self.assist_force_buf)
        )

        # # total_forces = damping_force + forces_lift
        total_forces = forces_lift
        
        # 接口要求的 shape: 
        # forces: (num_envs, num_bodies_to_apply, 3)
        # body_ids: (num_bodies_to_apply,)
        
        # 1. 调整 forces 维度: (num_envs, 3) -> (num_envs, 1, 3)
        forces_applied = total_forces.unsqueeze(1)
        
        # 2. 准备 torques (全0): (num_envs, 1, 3)
        # torques_applied = damping_torque.unsqueeze(1)
        torques_applied = torch.zeros_like(forces_applied)
        
        # 3. 指定 body_ids
        body_ids = torch.tensor([root_body_idx], device=self.device, dtype=torch.long)
        
        # 4. 调用接口写入 Buffer
        # 注意：这只是写入了 buffer，真正的施力发生在随后的 self.scene.write_data_to_sim()
        robot.set_external_force_and_torque(
            forces=forces_applied, 
            torques=torques_applied, 
            body_ids=body_ids,
            is_global=True
        )

        if self.sim.has_gui():
            # 将标记移动到机器人的 Root 位置
            # 我们可以给 Z 轴加一点偏移，让箭头浮在头顶上，不要插在肚子里
            marker_pos = robot.data.root_pos_w.clone()
            marker_pos[:, 2] += 0.6  # 向上偏移 0.5m

             # 获取要可视化的力向量 (num_envs, 3)
            vis_force = total_forces.clone()

            # 计算力的大小 (num_envs, 1)
            force_mag = torch.norm(vis_force, dim=-1, keepdim=True)  # (num_envs, 1)

            # 计算缩放比例 (简单线性缩放)
            visual_gain = 0.02 
            arrow_length = force_mag * visual_gain
            arrow_thickness = 0.8 

            # 组合 Scale: (num_envs, 3) -> [thickness, thickness, length]
            # 这里假设你的 marker 模型默认是沿 Z 轴竖立的圆柱/箭头
            target_scales = torch.cat([
                torch.full_like(arrow_length, arrow_thickness), # X scale (粗细)
                torch.full_like(arrow_length, arrow_thickness), # Y scale (粗细)
                arrow_length                                    # Z scale (长度，随力变化)
            ], dim=1)

            # 根据力的存在与否决定是否显示箭头
            is_visible = force_mag > 1e-3
            marker_scales = torch.where(
                is_visible, 
                target_scales, 
                torch.zeros_like(target_scales)
            )
            
            # 如果将来加入了水平阻尼力(damping)，箭头需要指向力的方向，
            # 则需要计算从 (0,0,1) 到 vis_force 的旋转四元数并传入 orientations 参数。
            self.assist_force_marker.visualize(
                translations=marker_pos,
                scales=marker_scales,
            )

    def _update_curriculum(self, env_ids: torch.Tensor):
        # 1. 获取重置环境的机器人的当前高度
        # 注意：必须在 super()._reset_idx() 之前调用，否则位置就被重置回起点了！
        robot = self.scene["robot"]
        root_height = robot.data.root_pos_w[env_ids, 2]

        # 2. 判定成功/失败
        success_thresh = self.cfg.target_stand_height
        is_success = root_height >= success_thresh
        is_failure = ~is_success

        # 注意：is_success / is_failure 是相对于 env_ids 的 boolean，需要映射回全局索引
        success_env_ids = env_ids[is_success]
        failure_env_ids = env_ids[is_failure]

        # === 成功：减少辅助力 / 动作缩放（降低难度辅助，让机器人更独立）===
        if len(success_env_ids) > 0:
            decay_force = self.cfg.force_decay_step
            min_force = self.cfg.min_assist_force
            self.assist_force_buf[success_env_ids] = torch.clamp(
                self.assist_force_buf[success_env_ids] - decay_force,
                min=min_force
            )

            decay_scale = self.cfg.scale_decay_step
            min_scale = self.cfg.min_action_scale
            # 注意：action_scale_buf 是 (num_envs, 1)，所以索引要匹配
            self.action_scale_buf[success_env_ids] = torch.clamp(
                self.action_scale_buf[success_env_ids] - decay_scale,
                min=min_scale
            )

        # === 失败：恢复辅助力 / 动作缩放（增大辅助，让机器人重新获得帮助）===
        if len(failure_env_ids) > 0:
            recover_force = self.cfg.force_recover_step
            max_force = self.cfg.initial_assist_force
            self.assist_force_buf[failure_env_ids] = torch.clamp(
                self.assist_force_buf[failure_env_ids] + recover_force,
                max=max_force
            )

            recover_scale = self.cfg.scale_recover_step
            max_scale = self.cfg.initial_action_scale
            self.action_scale_buf[failure_env_ids] = torch.clamp(
                self.action_scale_buf[failure_env_ids] + recover_scale,
                max=max_scale
            )

        # === 节流打印：每 200 次全局 step 仅打印一次统计摘要 ===
        if self.common_step_counter % 200 == 0:
            avg_force = self.assist_force_buf.mean().item()
            avg_scale = self.action_scale_buf.mean().item()
            n_success = len(success_env_ids)
            n_failure = len(failure_env_ids)
            print(
                f"[Curriculum] step={self.common_step_counter} | "
                f"reset: {n_success} success / {n_failure} fail | "
                f"avg_force={avg_force:.1f} | avg_scale={avg_scale:.3f}"
            )