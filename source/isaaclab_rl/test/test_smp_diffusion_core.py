# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import torch


def _load_diffusion_module(module_name: str):
    # 通过父目录回溯定位 diffusion 源码模块，避免依赖固定执行路径。
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "rsl_rl" / "rsl_rl" / "diffusion" / f"{module_name}.py"
        if module_path.exists():
            # 直接按文件路径动态导入，便于单元测试精确加载目标实现。
            spec = importlib.util.spec_from_file_location(f"isaaclab_smp_{module_name}_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Could not find rsl_rl/rsl_rl/diffusion/{module_name}.py")


def test_q_sample_matches_closed_form():
    # 验证前向扩散采样与理论闭式解一致。
    scheduler_module = _load_diffusion_module("scheduler")
    scheduler = scheduler_module.DiffusionScheduler(num_steps=50, beta_start=1e-4, beta_end=2e-2)
    x0 = torch.randn(4, 10, 131)
    eps = torch.randn_like(x0)
    t = torch.tensor([22, 15, 8, 22], dtype=torch.long)

    xt = scheduler.q_sample(x0, t, eps)

    alpha_bar = scheduler.alpha_bar[t].view(-1, 1, 1)
    expected = alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * eps
    assert torch.allclose(xt, expected)


def test_motion_epsilon_transformer_accepts_style_ids():
    # 条件模型需要接受 style_id，并保持输出形状与输入一致。
    model_module = _load_diffusion_module("model")
    model = model_module.MotionEpsilonTransformer(
        feature_dim=131,
        window_size=10,
        num_diffusion_steps=50,
        num_styles=4,
        hidden_dim=64,
        num_layers=1,
        num_heads=4,
    )
    xt = torch.randn(2, 10, 131)
    t = torch.tensor([22, 15], dtype=torch.long)
    style_id = torch.tensor([1, 3], dtype=torch.long)

    eps_hat = model(xt, t, style_id=style_id)

    assert eps_hat.shape == xt.shape


def test_conditioning_supports_null_style_dropout_and_cfg():
    # 条件模块需要同时覆盖 null-style、dropout 和 CFG 组合公式。
    conditioning_module = _load_diffusion_module("conditioning")
    style_id = torch.tensor([0, 2, 1], dtype=torch.long)

    dropped = conditioning_module.maybe_drop_style(
        style_id,
        drop_prob=1.0,
        null_style_id=conditioning_module.NULL_STYLE_ID,
    )
    cfg = conditioning_module.apply_classifier_free_guidance(
        eps_uncond=torch.tensor([[[1.0, 1.0]]]),
        eps_cond=torch.tensor([[[3.0, 5.0]]]),
        guidance_scale=1.5,
    )

    assert torch.equal(
        dropped,
        torch.full_like(style_id, conditioning_module.NULL_STYLE_ID),
    )
    assert torch.allclose(cfg, torch.tensor([[[4.0, 7.0]]]))
