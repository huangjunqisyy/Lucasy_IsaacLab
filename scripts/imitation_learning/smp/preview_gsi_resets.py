# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to preview SMP GSI reset states in Isaac Sim."""

"""Launch Isaac Sim Simulator first."""


import argparse
import contextlib
import sys

from path_bootstrap import find_nested_rsl_rl_repo_root, prepend_local_source_paths

# 先把当前 worktree 的本地 Isaac Lab 包前置到 sys.path，避免误用主工作区安装版本。
prepend_local_source_paths(__file__)
_RSL_RL_REPO_ROOT = find_nested_rsl_rl_repo_root(__file__)
if str(_RSL_RL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="在 Isaac Sim 里手动预览 SMP GSI 生成的 reset state。")
parser.add_argument("--task", type=str, required=True, help="任务名称，例如 Isaac-Velocity-Flat-G1-v0。")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="agent 配置入口，默认读取任务注册的 RSL-RL 配置。",
)
parser.add_argument("--num_envs", type=int, default=1, help="预览环境数，默认单环境。")
parser.add_argument("--seed", type=int, default=None, help="可选随机种子。")
parser.add_argument("--reset_key", type=str, default="N", help="触发下一次 GSI reset 采样的按键。")
parser.add_argument("--quit_key", type=str, default="ESCAPE", help="退出预览脚本的按键。")
parser.add_argument(
    "--sample_on_start",
    action="store_true",
    default=False,
    help="启动后立即触发一次 GSI reset 采样。",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import build_smp_preview_runtime, format_preview_reset_summary, sample_and_apply_preview_reset

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def _set_preview_camera(env, asset_name: str = "robot") -> None:
    # 将相机放到机器人斜后上方，方便观察每次 reset 后的站姿和速度趋势。
    target_env = getattr(env, "unwrapped", env)
    asset = target_env.scene[asset_name]
    root_pos = asset.data.root_pos_w[0].detach().cpu()
    eye = (root_pos + torch.tensor([2.5, -2.5, 1.5], dtype=root_pos.dtype)).tolist()
    target = (root_pos + torch.tensor([0.0, 0.0, 0.8], dtype=root_pos.dtype)).tolist()
    target_env.sim.set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    # 预览模式固定使用少量环境，避免多个实例同时刷新造成观察困难。
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.seed is not None:
        env_cfg.seed = int(args_cli.seed)
        if hasattr(agent_cfg, "seed"):
            agent_cfg.seed = int(args_cli.seed)
    if hasattr(env_cfg, "recorders"):
        env_cfg.recorders = {}
    if hasattr(env_cfg, "terminations"):
        env_cfg.terminations = {}

    env = gym.make(args_cli.task, cfg=env_cfg)
    runtime = build_smp_preview_runtime(agent_cfg=agent_cfg, device=env.unwrapped.device)
    zero_actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)

    state = {
        "pending_samples": 1 if args_cli.sample_on_start else 0,
        "quit_requested": False,
    }

    def _request_sample():
        state["pending_samples"] += 1

    def _request_quit():
        state["quit_requested"] = True

    keyboard = Se3Keyboard(
        Se3KeyboardCfg(
            gripper_term=False,
            pos_sensitivity=0.0,
            rot_sensitivity=0.0,
            sim_device=str(env.unwrapped.device),
        )
    )
    keyboard.add_callback(str(args_cli.reset_key).upper(), _request_sample)
    keyboard.add_callback(str(args_cli.quit_key).upper(), _request_quit)

    env.reset()
    _set_preview_camera(env, asset_name=runtime.asset_name)
    print(
        f'[SMP GSI Preview] 聚焦 Isaac Sim 视口后按 "{str(args_cli.reset_key).upper()}" 采样下一次 reset，'
        f'按 "{str(args_cli.quit_key).upper()}" 退出。'
    )
    print("[SMP GSI Preview] 当前展示的是 GSI 生成的初始状态，不是动作序列回放。")

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting() and not state["quit_requested"]:
            if state["pending_samples"] > 0:
                state["pending_samples"] -= 1
                reset_result = sample_and_apply_preview_reset(env=env, runtime=runtime, env_ids=env_ids)
                print(format_preview_reset_summary(reset_result))
                _set_preview_camera(env, asset_name=runtime.asset_name)
                env.unwrapped.sim.render()
            env.step(zero_actions)

    del keyboard
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
