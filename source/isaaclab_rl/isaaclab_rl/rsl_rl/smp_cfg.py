# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg


@configclass
class SMPPriorCfg:
    checkpoint_path: str = MISSING
    window_size: int = 10
    feature_dim: int = MISSING
    num_diffusion_steps: int = 50
    timesteps_k: list[int] = MISSING
    reward_scale: float = 1.0
    adaptive_norm_decay: float = 0.99
    log_histograms_every: int = 20


@configclass
class SMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type: str = "rsl_rl.runners:SMPOnPolicyRunner"
    smp_prior: SMPPriorCfg = MISSING
    smp_reward_coef: float = 1.0
    task_reward_coef: float = 1.0
