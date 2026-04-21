# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, field

from isaaclab.utils import configclass

from .rl_cfg import RslRlOnPolicyRunnerCfg


@configclass
class SMPStyleCfg:
    mode: str = "single_style"
    target_style_name: str = "walk"
    target_style_id: int | None = None
    guidance_scale: float = 1.0
    mask_name: str = "g1_upper_lower"
    shared_style_name: str | None = None
    body_part_style_names: dict[str, str] = field(default_factory=lambda: {"upper_body": "a", "lower_body": "c"})
    feature_block_offsets: dict[str, tuple[int, int]] = field(default_factory=dict)
    joint_name_order: list[str] = field(default_factory=list)
    joint_axes: list[tuple[float, float, float]] = field(default_factory=list)
    ee_name_order: list[str] = field(default_factory=list)
    key_body_name_order: list[str] = field(default_factory=list)


@configclass
class SMPGSICfg:
    enabled: bool = False
    sample_on_reset: bool = True
    guidance_scale: float | None = None
    fallback_to_default_reset: bool = True
    max_resample_attempts: int = 3
    error_threshold: float = 1.0e-6
    asset_name: str = "robot"


@configclass
class SMPPriorCfg:
    checkpoint_path: str = MISSING
    window_size: int = 10
    feature_dim: int = MISSING
    num_diffusion_steps: int = 50
    timesteps_k: list[int] = MISSING
    reward_mode: str = "absolute"
    reward_scale: float = 1.0
    adaptive_norm_decay: float = 0.999
    norm_anchor_path: str | None = None
    log_histograms_every: int = 20
    style_cfg: SMPStyleCfg = field(default_factory=SMPStyleCfg)


@configclass
class SMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type: str = "rsl_rl.runners:SMPOnPolicyRunner"
    smp_prior: SMPPriorCfg = MISSING
    smp_reward_coef: float = 1.0
    task_reward_coef: float = 1.0
    gsi_cfg: SMPGSICfg = field(default_factory=SMPGSICfg)
