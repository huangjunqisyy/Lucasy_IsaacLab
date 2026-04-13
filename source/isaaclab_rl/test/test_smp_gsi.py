# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib

import pytest
import torch


def _feature_layout(module):
    return module.SMPFeatureLayout.from_feature_block_offsets(
        {
            "base_lin_vel_b": (0, 3),
            "base_ang_vel_b": (3, 6),
            "joint_rot6d_rel": (6, 180),
            "ee_pos_b": (180, 192),
        }
    )


def test_gsi_decoder_uses_last_frame_and_reference_defaults():
    gsi_module = importlib.import_module("rsl_rl.diffusion.gsi")
    layout = _feature_layout(gsi_module)
    decoder = gsi_module.SMPGSIDecoder(
        feature_layout=layout,
        joint_axes=torch.tensor([[0.0, 1.0, 0.0]] * 29, dtype=torch.float32),
        error_threshold=1.0e-8,
    )

    window = torch.zeros(2, 10, 192)
    window[0, -1, 0:3] = torch.tensor([0.1, 0.2, 0.3])
    window[0, -1, 3:6] = torch.tensor([0.4, 0.5, 0.6])
    window[0, -1, 6:180] = _joint_rot6d_block(torch.linspace(-1.4, 1.4, 29, dtype=torch.float32)).reshape(-1)
    window[1, -1, 0:3] = torch.tensor([-0.2, 0.0, 0.5])
    window[1, -1, 3:6] = torch.tensor([0.7, -0.1, 0.2])
    window[1, -1, 6:180] = _joint_rot6d_block(torch.full((29,), 0.5, dtype=torch.float32)).reshape(-1)

    result = decoder.decode(
        window,
        reference_state=gsi_module.SMPResetReference(
            root_pos_w=torch.tensor([0.0, 0.0, 0.74]),
            root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            joint_pos=torch.zeros(29),
            joint_vel=torch.full((29,), 0.25),
        ),
    )

    assert result.supports_reset_state is True
    assert result.reconstruction_mse == pytest.approx(0.0)
    assert result.unrecoverable_feature_blocks == ("ee_pos_b",)
    assert torch.allclose(result.state.root_pos_w, torch.tensor([[0.0, 0.0, 0.74], [0.0, 0.0, 0.74]]))
    assert torch.allclose(result.state.root_lin_vel_w[0], torch.tensor([0.1, 0.2, 0.3]))
    assert torch.allclose(result.state.root_ang_vel_w[1], torch.tensor([0.7, -0.1, 0.2]))
    assert torch.allclose(result.state.joint_pos[0], torch.linspace(-1.4, 1.4, 29, dtype=torch.float32))
    assert torch.allclose(result.state.joint_vel[1], torch.full((29,), 0.25))


def test_gsi_decoder_rejects_unexpected_feature_dim():
    gsi_module = importlib.import_module("rsl_rl.diffusion.gsi")
    layout = _feature_layout(gsi_module)
    decoder = gsi_module.SMPGSIDecoder(
        feature_layout=layout,
        joint_axes=torch.tensor([[0.0, 1.0, 0.0]] * 29, dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="Expected motion window feature dim 192"):
        decoder.decode(
            torch.zeros(1, 10, 191),
            reference_state=gsi_module.SMPResetReference(
                root_pos_w=torch.tensor([0.0, 0.0, 0.74]),
                root_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                joint_pos=torch.zeros(29),
            ),
        )


def _joint_rot6d_block(angles: torch.Tensor) -> torch.Tensor:
    cos_angle = torch.cos(angles)
    sin_angle = torch.sin(angles)
    return torch.stack(
        (
            cos_angle,
            torch.zeros_like(angles),
            torch.zeros_like(angles),
            torch.ones_like(angles),
            -sin_angle,
            torch.zeros_like(angles),
        ),
        dim=-1,
    )
