# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import torch


class _DummyStyleModel(torch.nn.Module):
    """返回与 style_id 对应常数噪声，便于验证采样器的条件拼接逻辑。"""

    def __init__(self, num_styles: int):
        super().__init__()
        self.num_styles = num_styles

    def forward(self, xt: torch.Tensor, t: torch.Tensor, style_id: torch.Tensor | None = None) -> torch.Tensor:
        if style_id is None:
            value = torch.zeros(xt.shape[0], device=xt.device, dtype=xt.dtype)
        else:
            value = style_id.to(device=xt.device, dtype=xt.dtype)
        return value.view(-1, 1, 1).expand_as(xt)


def test_ddpm_sampler_returns_motion_window_shape():
    sampler_module = importlib.import_module("rsl_rl.diffusion.sampler")
    sampler = sampler_module.SMPDiffusionSampler(
        model=_DummyStyleModel(num_styles=4),
        num_diffusion_steps=4,
        feature_dim=6,
        window_size=3,
        device="cpu",
    )

    sample = sampler.sample(batch_size=4, style_id=torch.tensor([1, 1, 2, 3], dtype=torch.long))

    assert sample.shape == (4, 3, 6)


def test_sampler_body_mask_style_program_composes_part_predictions():
    sampler_module = importlib.import_module("rsl_rl.diffusion.sampler")
    sampler = sampler_module.SMPDiffusionSampler(
        model=_DummyStyleModel(num_styles=3),
        num_diffusion_steps=4,
        feature_dim=4,
        window_size=1,
        device="cpu",
    )

    eps = sampler.predict_eps(
        xt=torch.zeros(1, 1, 4),
        t=torch.tensor([1], dtype=torch.long),
        style_program={
            "mode": "body_mask",
            "guidance_scale": 1.0,
            "part_style_ids": {"shared_body": 2, "lower_body": 2, "upper_body": 0},
            "feature_masks": {
                "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
                "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
            },
        },
    )

    assert torch.allclose(eps, torch.tensor([[[2.0, 2.0, 0.0, 0.0]]]))
