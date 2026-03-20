# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_smp_features_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "mdp"
        / "smp_features.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_smp_feature_utils_unit", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_quat_to_rot6d_returns_six_values_per_body():
    smp_features = _load_smp_features_module()
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    rot6d = smp_features.quat_to_rot6d(quat)

    assert rot6d.shape == (1, 1, 6)


def test_pack_smp_frame_features_has_expected_dim():
    smp_features = _load_smp_features_module()

    features = smp_features.pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 3),
        base_ang_vel_b=torch.zeros(2, 3),
        joint_pos_rel=torch.zeros(2, 29),
        ee_pos_b=torch.zeros(2, 4, 3),
        key_body_quat_b=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 14, dtype=torch.float32).view(1, 14, 4).repeat(2, 1, 1),
    )

    assert features.shape == (2, 131)


def test_pack_smp_frame_features_raises_for_unexpected_dim():
    smp_features = _load_smp_features_module()

    with pytest.raises(ValueError, match="Expected SMP feature dim 130, got 131"):
        smp_features.pack_smp_frame_features(
            base_lin_vel_b=torch.zeros(2, 3),
            base_ang_vel_b=torch.zeros(2, 3),
            joint_pos_rel=torch.zeros(2, 29),
            ee_pos_b=torch.zeros(2, 4, 3),
            key_body_quat_b=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 14, dtype=torch.float32).view(1, 14, 4).repeat(2, 1, 1),
            expected_feature_dim=130,
        )
