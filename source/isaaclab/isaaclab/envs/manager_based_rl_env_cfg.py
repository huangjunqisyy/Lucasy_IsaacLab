# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .manager_based_env_cfg import ManagerBasedEnvCfg
from .ui import ManagerBasedRLEnvWindow


@configclass
class ManagerBasedRLEnvCfg(ManagerBasedEnvCfg):
    """Configuration for a reinforcement learning environment with the manager-based workflow."""

    # ui settings
    ui_window_class_type: type | None = ManagerBasedRLEnvWindow

    # general settings
    is_finite_horizon: bool = False
    """Whether the learning task is treated as a finite or infinite horizon problem for the agent.
    Defaults to False, which means the task is treated as an infinite horizon problem.

    This flag handles the subtleties of finite and infinite horizon tasks:

    * **Finite horizon**: no penalty or bootstrapping value is required by the the agent for
      running out of time. However, the environment still needs to terminate the episode after the
      time limit is reached.
    * **Infinite horizon**: the agent needs to bootstrap the value of the state at the end of the episode.
      This is done by sending a time-limit (or truncated) done signal to the agent, which triggers this
      bootstrapping calculation.

    If True, then the environment is treated as a finite horizon problem and no time-out (or truncated) done signal
    is sent to the agent. If False, then the environment is treated as an infinite horizon problem and a time-out
    (or truncated) done signal is sent to the agent.

    Note:
        The base :class:`ManagerBasedRLEnv` class does not use this flag directly. It is used by the environment
        wrappers to determine what type of done signal to send to the corresponding learning agent.
    """

    episode_length_s: float = MISSING
    """Duration of an episode (in seconds).

    Based on the decimation rate and physics time step, the episode length is calculated as:

    .. code-block:: python

        episode_length_steps = ceil(episode_length_s / (decimation_rate * physics_time_step))

    For example, if the decimation rate is 10, the physics time step is 0.01, and the episode length is 10 seconds,
    then the episode length in steps is 100.
    """

    # environment settings
    rewards: object = MISSING
    """Reward settings.

    Please refer to the :class:`isaaclab.managers.RewardManager` class for more details.
    """

    terminations: object = MISSING
    """Termination settings.

    Please refer to the :class:`isaaclab.managers.TerminationManager` class for more details.
    """

    curriculum: object | None = None
    """Curriculum settings. Defaults to None, in which case no curriculum is applied.

    Please refer to the :class:`isaaclab.managers.CurriculumManager` class for more details.
    """

    commands: object | None = None
    """Command settings. Defaults to None, in which case no commands are generated.

    Please refer to the :class:`isaaclab.managers.CommandManager` class for more details.
    """

    # ---------------------------------------------------------------------------
    # Get-Up Curriculum settings (disabled by default)
    # ---------------------------------------------------------------------------
    getup_curriculum: bool = True
    """Whether to enable the assistive get-up curriculum. Defaults to False."""

    initial_assist_force: float = 300.0
    """Initial upward assistive force (N) applied to the robot's pelvis. Defaults to 200.0."""

    min_assist_force: float = 0.0
    """Minimum assistive force after full curriculum progression. Defaults to 0.0."""

    force_decay_step: float = 10.0
    """Amount to decrease assist force per successful episode. Defaults to 10.0."""

    force_recover_step: float = 5.0
    """Amount to increase assist force per failed episode. Defaults to 5.0."""

    initial_action_scale: float = 4.0
    """Initial action scale multiplier applied per environment. Defaults to 4.0."""

    min_action_scale: float = 1.0
    """Minimum action scale after full curriculum progression. Defaults to 1.0."""

    scale_decay_step: float = 0.05
    """Amount to decrease action scale per successful episode. Defaults to 0.0."""

    scale_recover_step: float = 0.03
    """Amount to increase action scale per failed episode. Defaults to 0.0."""

    upright_threshold: float = 0.6
    """Z-component threshold of the robot's up-vector to be considered upright. Defaults to 0.7."""

    target_stand_height: float = 0.7
    """Root height (m) required to count an episode as a standing success. Defaults to 0.7."""
