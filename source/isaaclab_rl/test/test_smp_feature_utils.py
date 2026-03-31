# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import math
from pathlib import Path

import pytest
import torch


def _load_smp_features_module():
    # 直接按文件路径加载特征工具模块，避免测试依赖安装路径。
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
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _identity_key_body_quat(*lead_shape: int, num_bodies: int = 14) -> torch.Tensor:
    # 生成单位四元数输入，作为旋转相关特征测试的稳定基线。
    quat = torch.zeros(*lead_shape, num_bodies, 4, dtype=torch.float32)
    quat[..., 0] = 1.0
    return quat


def test_quat_to_rot6d_returns_six_values_per_body():
    # 验证 quat->rot6d 转换后每个刚体输出 6 维表示。
    smp_features = _load_smp_features_module()
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)

    rot6d = smp_features.quat_to_rot6d(quat)

    assert rot6d.shape == (1, 1, 6)
    assert torch.allclose(rot6d, torch.tensor([[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]]))


def test_quat_to_rot6d_preserves_row_major_first_two_columns_order():
    # 验证 rot6d 的展开顺序与约定一致（行优先的前两列）。
    smp_features = _load_smp_features_module()
    half_sqrt = math.sqrt(0.5)
    quat = torch.tensor([[[half_sqrt, 0.0, 0.0, half_sqrt]]], dtype=torch.float32)

    rot6d = smp_features.quat_to_rot6d(quat)

    expected = torch.tensor([[[0.0, -1.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    assert torch.allclose(rot6d, expected, atol=1e-5)


def test_pack_smp_frame_features_has_expected_dim():
    # 验证单帧特征打包后的最终维度与设计值一致。
    smp_features = _load_smp_features_module()

    features = smp_features.pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 3),
        base_ang_vel_b=torch.zeros(2, 3),
        joint_pos_rel=torch.zeros(2, 29),
        ee_pos_b=torch.zeros(2, 4, 3),
        key_body_quat_b=_identity_key_body_quat(2),
    )

    assert features.shape == (2, 131)


def test_pack_smp_frame_features_raises_for_unexpected_dim():
    # 维度约束回归：当期望维度不匹配时应抛出明确异常。
    smp_features = _load_smp_features_module()

    with pytest.raises(ValueError, match="Expected SMP feature dim 130, got 131"):
        smp_features.pack_smp_frame_features(
            base_lin_vel_b=torch.zeros(2, 3),
            base_ang_vel_b=torch.zeros(2, 3),
            joint_pos_rel=torch.zeros(2, 29),
            ee_pos_b=torch.zeros(2, 4, 3),
            key_body_quat_b=_identity_key_body_quat(2),
            expected_feature_dim=130,
        )


def test_pack_smp_frame_features_supports_multiple_leading_dims():
    # 验证函数可处理额外前导维（如 batch+time）并保持末维为特征维。
    smp_features = _load_smp_features_module()

    features = smp_features.pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 5, 3),
        base_ang_vel_b=torch.zeros(2, 5, 3),
        joint_pos_rel=torch.zeros(2, 5, 29),
        ee_pos_b=torch.zeros(2, 5, 4, 3),
        key_body_quat_b=_identity_key_body_quat(2, 5),
        expected_feature_dim=131,
    )

    assert features.shape == (2, 5, 131)
