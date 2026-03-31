# SMP Diffusion 端到端流程整理

这份文档把当前 `G1 + SMP diffusion prior` 的完整实现链路整理成一条可追踪的代码地图。

目标不是只列文件名，而是回答这几个问题：

- 原始运动数据从哪里来，如何被转成训练用特征
- diffusion prior 预训练时实际走了哪些类和函数
- 在线 RL 时哪个任务在调用哪个 runner
- frozen prior 是怎么被加载并转成 `SMP reward` 的
- TensorBoard 里每一类指标是从哪里写出去的

## 0. 仓库布局

这套实现跨了两个代码根目录：

- IsaacLab worktree：
  - `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion`
- 嵌套的独立 `rsl_rl` 仓库：
  - `/home/lucas/isaac-sim/IsaacLab/rsl_rl`

要特别注意：

- `任务配置`、`观测特征`、`训练入口脚本` 在 worktree
- `diffusion 核心`、`SMP reward`、`SMP runner` 在嵌套 `rsl_rl` 仓库

## 1. 一张总览图

```text
原始 G1 motion.npz
  -> export_g1_motion_dataset.py
    -> _build_smp_frames_from_raw_motion()
      -> pack_smp_frame_features()
  -> 导出 frames 数据集 npz
  -> SMPMotionWindowDataset
  -> train_motion_prior.py
    -> SMPDiffusionTrainer
      -> DiffusionScheduler
      -> MotionEpsilonTransformer
      -> log_smp_pretrain_metrics()
  -> model_latest.pt
  -> G1SMPRunnerCfg / SMPRunnerCfg
  -> train.py
    -> hydra_task_config()
    -> resolve_runner_class()
    -> SMPOnPolicyRunner
      -> _load_prior_model()
      -> env obs["smp_motion_window"]
        -> smp_frame_features()
          -> pack_smp_frame_features()
      -> _compute_smp_metrics()
        -> DiffusionScheduler.q_sample()
        -> MotionEpsilonTransformer(xt, t)
        -> SMPReward.compute()
      -> _combine_rewards()
  -> 总 reward = task_reward_coef * task_reward + smp_reward_coef * smp_reward
  -> TensorBoard: SMP/reward, SMP/noise_mse, SMP/t*/noise_mse
```

## 2. 特征定义层

这一层的职责是定义“单帧 SMP 特征长什么样”，并保证：

- 离线导出时用的是同一套特征拼接顺序
- 在线环境观测时用的是同一套特征拼接顺序

### 2.1 常量配置

文件：

- [config.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py)

关键内容：

- `g1_smp_joint_names`
- `g1_ee_names`
- `g1_key_body_names`
- `g1_smp_window_size`
- `g1_smp_feature_dim`
- `g1_smp_num_diffusion_steps`
- `g1_smp_timesteps_k`

这份文件定义了所有后续模块共享的维度和 body/joint 顺序。

### 2.2 单帧特征拼接

文件：

- [smp_features.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py)

关键函数：

- `quat_to_rot6d()`
- `pack_smp_frame_features()`

`pack_smp_frame_features()` 的特征顺序固定为：

1. `base_lin_vel_b`
2. `base_ang_vel_b`
3. `joint_pos_rel`
4. `ee_pos_b`
5. `key_body_quat_b -> rot6d`

这是整个链路里最核心的“数据协议”。  
离线导出和在线观测必须共用它，否则 prior 训练和 RL 运行看到的就不是同一种输入。

### 2.3 在线观测提取

文件：

- [my_observations.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py)
- [velocity_env_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

关键调用链：

```text
SmpMotionWindowCfg.motion_frame
  -> mdp.smp_frame_features(env, ...)
    -> 从 Articulation 取:
       root_lin_vel_b
       root_ang_vel_b
       joint_pos - default_joint_pos
       ee_pos_w -> ee_pos_b
       key_body_quat_w -> key_body_quat_b
    -> pack_smp_frame_features(...)
```

在线环境里最终给 runner 的观测组名是：

- `smp_motion_window`

并且在 `SmpMotionWindowCfg` 中设置了：

- `history_length = g1_smp_window_size`
- `flatten_history_dim = True`

所以 runner 拿到的是扁平后的 `[B, T * F]`，随后会自己还原成 `[B, T, F]`。

## 3. 原始运动数据 -> SMP 训练数据集

这一层负责把原始 `motion.npz` 变成可训练的 `frames` 数据集。

### 3.1 导出脚本

文件：

- [export_g1_motion_dataset.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/export_g1_motion_dataset.py)

关键入口：

- `main()`
- `export_g1_motion_dataset()`

关键调用链：

```text
main()
  -> export_g1_motion_dataset(input_path, output_path, ...)
    -> _load_motion_arrays()
    -> 如果输入里已有 frames:
         直接使用
       否则:
         _build_smp_frames_from_raw_motion()
           -> 读取 raw motion:
              joint_pos
              body_pos_w
              body_quat_w
              body_lin_vel_w
              body_ang_vel_w
           -> 计算根坐标系下的:
              base_lin_vel_b
              base_ang_vel_b
              joint_pos_rel
              ee_pos_b
              key_body_quat_b
           -> pack_smp_frame_features()
    -> np.savez(...)
```

输入原始 `npz` 需要的关键键：

- `fps`
- `joint_pos`
- `body_pos_w`
- `body_quat_w`
- `body_lin_vel_w`
- `body_ang_vel_w`

输出数据集 `npz` 会写入：

- `frames`
- `fps`
- `window_size`
- `stride`
- `feature_dim`
- `joint_names`
- `ee_names`
- `key_body_names`

### 3.2 训练数据集封装

文件：

- [smp_dataset.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/smp_dataset.py)
- [motion/__init__.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/__init__.py)

关键类：

- `SMPMotionWindowDataset`

关键调用链：

```text
SMPMotionWindowDataset(path, window_size, stride)
  -> 读取 frames
  -> 根据 window_size / stride 构建 start_ids
  -> __getitem__(index)
     -> 返回 frames[start : start + window_size]
```

也就是说：

- 导出脚本产出的 `frames` 是单帧特征序列
- 真正训练 diffusion 时，`SMPMotionWindowDataset` 才把它切成滑动窗口

## 4. diffusion prior 离线预训练

### 4.1 训练脚本入口

文件：

- [train_motion_prior.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/train_motion_prior.py)

关键入口：

- `main()`

关键调用链：

```text
main()
  -> SMPDiffusionTrainer(...)
  -> trainer.train()
```

### 4.2 Trainer 主体

文件：

- [trainer.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/trainer.py)

关键类：

- `SMPDiffusionTrainer`

初始化阶段：

```text
SMPDiffusionTrainer.__init__()
  -> SMPMotionWindowDataset(...)
  -> DataLoader(...)
  -> DiffusionScheduler(...)
  -> MotionEpsilonTransformer(...)
  -> ExponentialMovingAverage(...)
  -> AdamW(...)
  -> SummaryWriter(...)
```

训练阶段：

```text
train()
  for global_step in [1 .. max_iters]:
    -> _next_batch()
    -> _compute_loss(x0)
      -> scheduler.sample_timesteps(...)
      -> eps = randn_like(x0)
      -> xt = scheduler.q_sample(x0, t, eps)
      -> eps_hat = model(xt, t)
      -> sample_mse = mean((eps_hat - eps)^2)
      -> loss = sample_mse.mean()
    -> optimizer.step()
    -> ema.update(model)
    -> log_smp_pretrain_metrics(...)
  -> save_checkpoint()
```

### 4.3 扩散核心组件

文件：

- [scheduler.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/scheduler.py)
- [model.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/model.py)
- [ema.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/ema.py)

职责分工：

- `DiffusionScheduler`
  - 维护 `beta / alpha / alpha_bar`
  - 提供 `sample_timesteps()`
  - 提供 `q_sample(x0, t, eps)`
- `MotionEpsilonTransformer`
  - 输入：`xt` 和离散时间步 `t`
  - 输出：预测噪声 `eps_hat`
- `ExponentialMovingAverage`
  - 维护训练期 EMA shadow weights

### 4.4 预训练日志

文件：

- [logging.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/logging.py)

预训练阶段写出的 TensorBoard tag：

- `SMPPretrain/loss`
- `SMPPretrain/noise_mse`
- `SMPPretrain/t22/noise_mse`
- `SMPPretrain/t15/noise_mse`
- `SMPPretrain/t8/noise_mse`
- `SMPPretrain/eps`
- `SMPPretrain/eps_hat`
- `SMPPretrain/eps_gap`

### 4.5 checkpoint 内容

`SMPDiffusionTrainer.save_checkpoint()` 会保存：

- `model_state_dict`
- `ema_state_dict`
- `optimizer_state_dict`
- `feature_dim`
- `window_size`
- `timesteps_k`
- `model_cfg`

产物通常是：

- `logs/smp_prior/g1/pretrain/model_latest.pt`

## 5. 在线任务入口与 runner 解析链

这一层决定“训练脚本最终调用的是哪个 runner”。

### 5.1 任务注册

文件：

- [g1/__init__.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py)

这里注册了任务：

- `Isaac-SMP-Velocity-Flat-G1-v0`

并把它的 `rsl_rl_cfg_entry_point` 指向：

- `G1SMPRunnerCfg`

### 5.2 Runner 配置

文件：

- [rsl_rl_ppo_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py)
- [smp_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py)

关键类：

- `G1SMPRunnerCfg`
- `SMPPriorCfg`
- `SMPRunnerCfg`

关键配置链：

```text
G1SMPRunnerCfg
  -> 继承 SMPRunnerCfg
  -> smp_prior = SMPPriorCfg(...)
  -> smp_reward_coef
  -> task_reward_coef

SMPRunnerCfg
  -> runner_type = "rsl_rl.runners:SMPOnPolicyRunner"
```

当前实现里，`G1SMPRunnerCfg` 会提供：

- prior checkpoint 路径
- feature dim
- window size
- diffusion steps
- `timesteps_k`
- `reward_scale`
- `smp_reward_coef`
- `task_reward_coef`

### 5.3 train.py 到 runner 的解析链

文件：

- [train.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/reinforcement_learning/rsl_rl/train.py)
- [hydra.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py)
- [runner_factory.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/isaaclab_rl/rsl_rl/runner_factory.py)

实际代码链：

```text
train.py
  -> @hydra_task_config(args_cli.task, args_cli.agent)
    -> register_task_to_hydra(task_name, agent_cfg_entry_point)
      -> load_cfg_from_registry(task_name, "env_cfg_entry_point")
      -> load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
  -> main(env_cfg, agent_cfg)
    -> env = gym.make(task, cfg=env_cfg, ...)
    -> env = RslRlVecEnvWrapper(env, ...)
    -> runner_class = resolve_runner_class(agent_cfg)
    -> runner = runner_class(...)
    -> runner.learn(...)
```

`resolve_runner_class(agent_cfg)` 在这里会把：

- `runner_type = "rsl_rl.runners:SMPOnPolicyRunner"`

解析成真正的 Python 类：

- `rsl_rl.runners.SMPOnPolicyRunner`

## 6. frozen prior 在线打分并生成 SMP reward

这一层是“预训练模型如何在 RL 里变成奖励”的核心。

### 6.1 在线 runner

文件：

- [smp_on_policy_runner.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/smp_on_policy_runner.py)
- [runners/__init__.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/__init__.py)

关键类：

- `SMPOnPolicyRunner`

### 6.2 初始化阶段

`SMPOnPolicyRunner.__init__()` 会做这些事：

```text
super().__init__(...)
-> 读取 smp_prior 配置
-> 构建 DiffusionScheduler
-> 构建 SMPReward
-> _load_prior_model()
```

`_load_prior_model()` 的链路是：

```text
checkpoint = torch.load(checkpoint_path)
-> 读取 checkpoint["model_cfg"]
-> 重建 MotionEpsilonTransformer(...)
-> model.load_state_dict(checkpoint["model_state_dict"], strict=True)
-> model.eval()
-> parameter.requires_grad_(False)
```

也就是说，在线阶段的 prior 是：

- 从 checkpoint 重建出来的
- `eval()` 模式
- 全参数冻结

当前实现的一个细节是：

- 在线 runner 加载的是 `model_state_dict`
- 没有把 `ema_state_dict` 覆盖回模型

所以现在真正用于奖励的是“训练末尾模型权重”，不是 EMA 权重。

### 6.3 从在线观测到 SMP reward 的代码链

`SMPOnPolicyRunner.learn()` 里每个 environment step 的核心链路是：

```text
actions = self.alg.act(obs)
obs, rewards, dones, extras = self.env.step(actions)

smp_metrics = self._compute_smp_metrics(obs)
rewards = self._combine_rewards(rewards, smp_metrics["reward"])
self.alg.process_env_step(obs, rewards, dones, extras)
```

展开 `_compute_smp_metrics(obs)`：

```text
_restore_smp_window(obs)
  -> obs["smp_motion_window"]
  -> reshape [B, T*F] -> [B, T, F]

for timestep in timesteps_k:
  -> eps_t = randn_like(x0)
  -> xt = scheduler.q_sample(x0, t, eps_t)
  -> eps_hat_t = smp_prior(xt, t)

metrics = SMPReward.compute(eps, eps_hat)
```

再展开 `SMPReward.compute()`：

文件：

- [smp_reward.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/smp_reward.py)

计算链：

```text
对每个 timestep:
  mse_t = mean((eps_hat_t - eps_t)^2)
  normalized_mse_t = mse_t / running_mse_t

noise_mse = mean_t(normalized_mse_t)
reward = exp(-reward_scale * noise_mse)
```

最后奖励融合：

```text
total_reward =
    task_reward_coef * task_reward
  + smp_reward_coef  * smp_reward
```

注意：

- `SMP reward` 不是写在环境 `RewardManager` 里的 term
- 它是在 `runner` 里额外算出来，再与环境 reward 线性融合

## 7. TensorBoard 日志链

### 7.1 预训练阶段

文件：

- [logging.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/logging.py)
- [trainer.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/trainer.py)

调用链：

```text
SMPDiffusionTrainer.train()
  -> log_smp_pretrain_metrics(writer, ...)
```

### 7.2 在线训练阶段

文件：

- [smp_on_policy_runner.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/smp_on_policy_runner.py)
- [logging.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/logging.py)

调用链：

```text
SMPOnPolicyRunner.learn()
  -> self.log(...)
    -> writer.add_scalar("SMP/reward", ...)
    -> writer.add_scalar("SMP/noise_mse", ...)
    -> writer.add_scalar("SMP/t22/noise_mse", ...)
    -> writer.add_scalar("SMP/t15/noise_mse", ...)
    -> writer.add_scalar("SMP/t8/noise_mse", ...)
    -> 每隔 log_histograms_every:
         log_smp_noise_metrics(...)
```

在线阶段你能看到的标量 tag：

- `SMP/reward`
- `SMP/noise_mse`
- `SMP/t22/noise_mse`
- `SMP/t15/noise_mse`
- `SMP/t8/noise_mse`

在线阶段你能看到的直方图 tag：

- `SMP/eps_true_t22`
- `SMP/eps_pred_t22`
- `SMP/eps_gap_t22`
- `SMP/eps_true_t15`
- `SMP/eps_pred_t15`
- `SMP/eps_gap_t15`
- `SMP/eps_true_t8`
- `SMP/eps_pred_t8`
- `SMP/eps_gap_t8`

## 8. 端到端相关文件清单

### 8.1 特征与任务配置

- [config.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py)
- [smp_features.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py)
- [my_observations.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py)
- [velocity_env_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py)

### 8.2 离线数据导出

- [export_g1_motion_dataset.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/export_g1_motion_dataset.py)

### 8.3 数据集与 diffusion 预训练

- [smp_dataset.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/smp_dataset.py)
- [motion/__init__.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/__init__.py)
- [train_motion_prior.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/train_motion_prior.py)
- [trainer.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/trainer.py)
- [scheduler.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/scheduler.py)
- [model.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/model.py)
- [ema.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/ema.py)
- [logging.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/logging.py)
- [diffusion/__init__.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/__init__.py)

### 8.4 在线 SMP reward 与 runner

- [smp_reward.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/smp_reward.py)
- [smp_on_policy_runner.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/smp_on_policy_runner.py)
- [runners/__init__.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/__init__.py)

### 8.5 在线训练入口与 runner 解析

- [train.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/reinforcement_learning/rsl_rl/train.py)
- [g1/__init__.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py)
- [rsl_rl_ppo_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py)
- [smp_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py)
- [runner_factory.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/isaaclab_rl/rsl_rl/runner_factory.py)
- [hydra.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py)

### 8.6 测试与验证

- [test_smp_feature_utils.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_feature_utils.py)
- [test_smp_dataset.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_dataset.py)
- [test_smp_diffusion_core.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_diffusion_core.py)
- [test_smp_reward_logging.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_reward_logging.py)
- [test_hydra.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/test/test_hydra.py)
- [test_rsl_rl_runner_factory.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_rsl_rl_runner_factory.py)

## 9. 你读源码时的推荐顺序

如果你想最快理解整条链，建议按这个顺序读：

1. [config.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py)
2. [smp_features.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py)
3. [export_g1_motion_dataset.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/export_g1_motion_dataset.py)
4. [smp_dataset.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/smp_dataset.py)
5. [trainer.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/trainer.py)
6. [model.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/model.py)
7. [smp_reward.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/smp_reward.py)
8. [smp_on_policy_runner.py](/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/smp_on_policy_runner.py)
9. [rsl_rl_ppo_cfg.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py)
10. [train.py](/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/reinforcement_learning/rsl_rl/train.py)

这样读基本就是按“数据协议 -> 离线训练 -> 在线奖励 -> 训练入口”的方向走，不容易迷路。
