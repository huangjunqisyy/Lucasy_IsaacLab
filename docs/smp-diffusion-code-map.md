# SMP Diffusion 代码位置清单

这份文档用来说明当前这轮 SMP diffusion 相关实现分别写到了哪里。

## 目录说明

- 当前工作树目录：`/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion`
- 主项目根目录：`/home/lucas/isaac-sim/IsaacLab`
- 需要特别注意：`rsl_rl` 是嵌套的独立仓库，所以 diffusion 核心代码不在当前 worktree 里，而是在主项目根目录下的 `rsl_rl/`

## 1. 扩散核心实现

这些文件位于主项目根目录下的嵌套 `rsl_rl` 仓库：

- `rsl_rl/rsl_rl/diffusion/__init__.py`
- `rsl_rl/rsl_rl/diffusion/scheduler.py`
- `rsl_rl/rsl_rl/diffusion/model.py`
- `rsl_rl/rsl_rl/diffusion/ema.py`
- `rsl_rl/rsl_rl/diffusion/logging.py`
- `rsl_rl/rsl_rl/diffusion/trainer.py`
- `rsl_rl/rsl_rl/diffusion/smp_reward.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/__init__.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/scheduler.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/model.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/ema.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/logging.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/trainer.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/diffusion/smp_reward.py`

## 2. 在线 runner 集成

这些文件也位于嵌套 `rsl_rl` 仓库：

- `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- `rsl_rl/rsl_rl/runners/__init__.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/runners/__init__.py`

## 3. 离线数据导出与预训练脚本

这些文件位于当前 worktree：

- `scripts/imitation_learning/smp/export_g1_motion_dataset.py`
- `scripts/imitation_learning/smp/train_motion_prior.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/export_g1_motion_dataset.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/scripts/imitation_learning/smp/train_motion_prior.py`

## 4. 数据集加载器

数据集加载器位于嵌套 `rsl_rl` 仓库：

- `rsl_rl/rsl_rl/motion/smp_dataset.py`
- `rsl_rl/rsl_rl/motion/__init__.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/smp_dataset.py`
- `/home/lucas/isaac-sim/IsaacLab/rsl_rl/rsl_rl/motion/__init__.py`

## 5. G1 任务配置接线

这些文件位于当前 worktree：

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py`
- `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py`

## 6. 共享 SMP 特征与观测

这些文件位于当前 worktree：

- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`

## 7. 测试文件

这些测试位于当前 worktree：

- `source/isaaclab_rl/test/test_smp_feature_utils.py`
- `source/isaaclab_rl/test/test_smp_dataset.py`
- `source/isaaclab_rl/test/test_smp_diffusion_core.py`
- `source/isaaclab_rl/test/test_smp_reward_logging.py`

对应绝对路径：

- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_feature_utils.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_dataset.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_diffusion_core.py`
- `/home/lucas/isaac-sim/IsaacLab/.worktrees/g1-smp-diffusion/source/isaaclab_rl/test/test_smp_reward_logging.py`

## 8. 当前状态说明

- 能在纯 Python 环境里验证通过的 diffusion、dataset、logging 相关测试已经通过
- 离线 pretrain 脚本已经可以产出 TensorBoard tag，包括：
  - `SMPPretrain/noise_mse`
  - `SMPPretrain/t22/noise_mse`
  - `SMPPretrain/t15/noise_mse`
  - `SMPPretrain/t8/noise_mse`
  - `SMPPretrain/eps`
  - `SMPPretrain/eps_hat`
  - `SMPPretrain/eps_gap`
- 需要 IsaacLab 运行时参与的 Hydra / 在线训练入口，当前还受 worktree 与主仓库路径分离影响，验证还没完全收口
