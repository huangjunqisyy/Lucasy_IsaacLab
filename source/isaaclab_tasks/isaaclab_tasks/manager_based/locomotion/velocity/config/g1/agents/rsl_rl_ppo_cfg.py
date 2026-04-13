# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    AMPDataCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    SMPGSICfg,
    SMPPriorCfg,
    SMPRunnerCfg,
    SMPStyleCfg,
)

from .config import (
    g1_anchor_name,
    g1_ee_names,
    g1_key_body_names,
    g1_root_name,
    g1_smp_feature_block_offsets,
    g1_smp_feature_dim,
    g1_smp_joint_axes,
    g1_smp_joint_names,
    g1_smp_mask_template_name,
    g1_smp_num_diffusion_steps,
    g1_smp_timesteps_k,
    g1_smp_window_size,
)

@configclass
class G1RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "g1_rough"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1FlatPPORunnerCfg(G1RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 1500
        self.experiment_name = "g1_flat"
        self.policy.actor_hidden_dims = [256, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]


@configclass
class G1SMPRunnerCfg(SMPRunnerCfg):
    """G1 SMP locomotion runner 配置。"""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_smp"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    smp_prior = SMPPriorCfg(
        checkpoint_path="logs/smp_prior/g1/pretrain_407/model_latest.pt",
        feature_dim=g1_smp_feature_dim,
        window_size=g1_smp_window_size,
        num_diffusion_steps=g1_smp_num_diffusion_steps,
        timesteps_k=g1_smp_timesteps_k,
        reward_scale=1.0,
        style_cfg=SMPStyleCfg(
            mode="single_style",
            target_style_name="walk",
            guidance_scale=1.0,
            mask_name=g1_smp_mask_template_name,
            feature_block_offsets=g1_smp_feature_block_offsets,
            joint_name_order=g1_smp_joint_names,
            joint_axes=g1_smp_joint_axes,
            ee_name_order=g1_ee_names,
            key_body_name_order=g1_key_body_names,
        ),
    )
    smp_reward_coef = 0.2
    task_reward_coef = 1.0
    gsi_cfg = SMPGSICfg()

@configclass
class G1AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1_amp"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    amp_data = AMPDataCfg(
        motion_files=[
            "/home/lucas/isaac-sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/datasetswalking1.npz"
        ],
        body_names = g1_key_body_names,
        # root_name = g1_root_name[0],
        # ee_names = g1_ee_names,
        amp_obs_terms = [
            # "joint_pos", "joint_vel", "body_pos_b", "body_quat_b", "body_lin_vel_b", "body_ang_vel_b"
            "joint_pos", "joint_vel"
        ],
        anchor_name = g1_anchor_name,
        discriminator_lr = 1.0e-3,
        num_learning_epochs=5,
        num_mini_batches=4
    )
    amp_discr_hidden_dims = [256, 256]
    amp_reward_coef = 1.0
    amp_task_reward_lerp = 0.5
    # amp_cfg = dict(
    #     amp_obs_normalization=False,
    #     amp_hidden_dims=[256, 256, 256],
    #     activation="elu",
    #     state_dependent_std=False,
    # )
