# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path
import sys
import types

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter
import torch


def _load_diffusion_module(module_name: str):
    # 沿父目录回溯定位 diffusion 子模块，避免测试依赖固定 cwd。
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "rsl_rl" / "rsl_rl" / "diffusion" / f"{module_name}.py"
        if module_path.exists():
            # 通过文件路径动态导入目标模块，便于单元测试直接调用实现。
            spec = importlib.util.spec_from_file_location(f"isaaclab_smp_{module_name}_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Could not find rsl_rl/rsl_rl/diffusion/{module_name}.py")


def _load_smp_cfg_module():
    fake_isaaclab = types.ModuleType("isaaclab")
    fake_isaaclab.__path__ = []
    fake_utils = types.ModuleType("isaaclab.utils")
    fake_utils.configclass = lambda cls: cls

    fake_package = types.ModuleType("isaaclab_rl")
    fake_package.__path__ = []
    fake_subpackage = types.ModuleType("isaaclab_rl.rsl_rl")
    fake_subpackage.__path__ = []
    fake_rl_cfg = types.ModuleType("isaaclab_rl.rsl_rl.rl_cfg")
    fake_rl_cfg.RslRlOnPolicyRunnerCfg = object

    sys.modules.setdefault("isaaclab", fake_isaaclab)
    sys.modules.setdefault("isaaclab.utils", fake_utils)
    sys.modules.setdefault("isaaclab_rl", fake_package)
    sys.modules.setdefault("isaaclab_rl.rsl_rl", fake_subpackage)
    sys.modules.setdefault("isaaclab_rl.rsl_rl.rl_cfg", fake_rl_cfg)

    for parent in Path(__file__).resolve().parents:
        module_path = parent / "source" / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "smp_cfg.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_rl.rsl_rl.smp_cfg", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py")


def test_log_smp_pretrain_metrics_writes_noise_tags(tmp_path):
    # 验证预训练日志默认只写入噪声相关标量，不写大体积直方图标签。
    logging_module = _load_diffusion_module("logging")
    writer = SummaryWriter(log_dir=tmp_path)

    logging_module.log_smp_pretrain_metrics(
        writer,
        global_step=1,
        loss=0.5,
        per_timestep_mse={22: 0.6, 15: 0.4, 8: 0.3},
        eps=torch.zeros(8, 10, 131),
        eps_hat=torch.ones(8, 10, 131),
    )
    writer.flush()
    writer.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()

    assert "SMPPretrain/noise_mse" in accumulator.Tags()["scalars"]
    assert "SMPPretrain/loss_total" in accumulator.Tags()["scalars"]
    assert "SMPPretrain/t22/noise_mse" in accumulator.Tags()["scalars"]
    assert "SMPPretrain/eps_gap" not in accumulator.Tags()["histograms"]


def test_log_smp_pretrain_metrics_can_write_histograms_when_enabled(tmp_path):
    logging_module = _load_diffusion_module("logging")
    writer = SummaryWriter(log_dir=tmp_path)

    logging_module.log_smp_pretrain_metrics(
        writer,
        global_step=1,
        loss=0.5,
        per_timestep_mse={22: 0.6, 15: 0.4, 8: 0.3},
        eps=torch.zeros(8, 10, 131),
        eps_hat=torch.ones(8, 10, 131),
        write_histograms=True,
    )
    writer.flush()
    writer.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()

    assert "SMPPretrain/eps_gap" in accumulator.Tags()["histograms"]


def test_smp_reward_uses_fixed_timestep_ensemble():
    # 验证奖励计算使用固定时间步集合并返回逐时间步误差统计。
    reward_module = _load_diffusion_module("smp_reward")
    rewarder = reward_module.SMPReward(
        num_diffusion_steps=50, timesteps_k=[22, 15, 8], reward_scale=1.0, reward_mode="absolute"
    )
    eps = {
        22: torch.zeros(4, 10, 131),
        15: torch.zeros(4, 10, 131),
        8: torch.zeros(4, 10, 131),
    }
    eps_hat = {
        22: torch.ones(4, 10, 131),
        15: torch.ones(4, 10, 131),
        8: torch.ones(4, 10, 131),
    }

    out = rewarder.compute(eps=eps, eps_hat=eps_hat)

    assert out["reward"].shape == (4,)
    assert set(out["per_timestep_mse"].keys()) == {22, 15, 8}


def test_smp_reward_target_vs_uncond_rewards_target_advantage():
    reward_module = _load_diffusion_module("smp_reward")
    rewarder = reward_module.SMPReward(
        num_diffusion_steps=50, timesteps_k=[22, 15, 8], reward_scale=1.0, reward_mode="target_vs_uncond"
    )
    eps = {
        22: torch.zeros(4, 10, 131),
        15: torch.zeros(4, 10, 131),
        8: torch.zeros(4, 10, 131),
    }
    eps_hat_target = {
        22: torch.full((4, 10, 131), 0.5),
        15: torch.full((4, 10, 131), 0.5),
        8: torch.full((4, 10, 131), 0.5),
    }
    eps_hat_uncond = {
        22: torch.ones(4, 10, 131),
        15: torch.ones(4, 10, 131),
        8: torch.ones(4, 10, 131),
    }

    out = rewarder.compute(eps=eps, eps_hat=eps_hat_target, eps_hat_uncond=eps_hat_uncond)

    assert out["reward"].shape == (4,)
    reward_target = torch.exp(torch.tensor(-0.25))
    reward_uncond = torch.exp(torch.tensor(-1.0))
    expected_reward = ((reward_target - reward_uncond) / (1.0 - reward_uncond).clamp_min(1.0e-6)).clamp(0.0, 1.0)
    assert torch.allclose(out["reward"], torch.full((4,), expected_reward))
    assert out["noise_mse"].shape == (4,)


def test_smp_prior_cfg_defaults_to_absolute_reward_mode():
    cfg_module = _load_smp_cfg_module()

    assert cfg_module.SMPPriorCfg.reward_mode == "absolute"


def test_log_smp_noise_metrics_writes_timestep_histogram_tags(tmp_path):
    # 验证在线噪声日志按时间步写入对应标量与直方图标签。
    logging_module = _load_diffusion_module("logging")
    writer = SummaryWriter(log_dir=tmp_path)

    logging_module.log_smp_noise_metrics(
        writer,
        global_step=2,
        noise_mse=0.5,
        per_timestep_mse={22: 0.6, 15: 0.4, 8: 0.3},
        eps={
            22: torch.zeros(4, 10, 131),
            15: torch.zeros(4, 10, 131),
            8: torch.zeros(4, 10, 131),
        },
        eps_hat={
            22: torch.ones(4, 10, 131),
            15: torch.ones(4, 10, 131),
            8: torch.ones(4, 10, 131),
        },
        prefix="SMP",
    )
    writer.flush()
    writer.close()

    accumulator = EventAccumulator(str(tmp_path))
    accumulator.Reload()

    assert "SMP/noise_mse" in accumulator.Tags()["scalars"]
    assert "SMP/t22/noise_mse" in accumulator.Tags()["scalars"]
    assert "SMP/eps_true_t22" in accumulator.Tags()["histograms"]
    assert "SMP/eps_pred_t22" in accumulator.Tags()["histograms"]
    assert "SMP/eps_gap_t22" in accumulator.Tags()["histograms"]
