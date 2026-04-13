# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_smp_diagnostics_module():
    module_path = Path(__file__).resolve().parents[1] / "isaaclab_rl" / "rsl_rl" / "smp_diagnostics.py"
    spec = importlib.util.spec_from_file_location("isaaclab_smp_diagnostics_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smp_window_collector_saves_windows_with_metadata(tmp_path):
    diagnostics = _load_smp_diagnostics_module()
    collector = diagnostics.SMPWindowCollector(obs_group="smp_motion_window", window_size=3, feature_dim=4)

    collector.add({"smp_motion_window": torch.arange(24, dtype=torch.float32).view(2, 12)})
    collector.add({"smp_motion_window": (100 + torch.arange(24, dtype=torch.float32)).view(2, 12)})
    output_path = collector.save(tmp_path / "policy_windows.npz")

    with np.load(output_path) as data:
        assert data["windows"].shape == (4, 3, 4)
        assert data["step_index"].tolist() == [0, 0, 1, 1]
        assert data["env_id"].tolist() == [0, 1, 0, 1]
        assert int(data["window_size"][0]) == 3
        assert int(data["feature_dim"][0]) == 4
        assert str(data["obs_group"][0]) == "smp_motion_window"


def test_load_windows_supports_raw_frames_and_policy_window_files(tmp_path):
    diagnostics = _load_smp_diagnostics_module()

    raw_motion_path = tmp_path / "raw_motion.npz"
    body_quat_w = np.zeros((12, 30, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0
    np.savez(
        raw_motion_path,
        fps=np.array([30], dtype=np.int64),
        joint_pos=np.zeros((12, 29), dtype=np.float32),
        joint_vel=np.zeros((12, 29), dtype=np.float32),
        body_pos_w=np.zeros((12, 30, 3), dtype=np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((12, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((12, 30, 3), dtype=np.float32),
    )

    policy_windows_path = tmp_path / "policy_windows.npz"
    np.savez(
        policy_windows_path,
        windows=np.zeros((5, 10, 192), dtype=np.float32),
        window_size=np.array([10], dtype=np.int64),
        feature_dim=np.array([192], dtype=np.int64),
        obs_group=np.asarray(["smp_motion_window"], dtype=np.str_),
    )

    raw_windows, raw_meta = diagnostics.load_motion_windows(
        raw_motion_path,
        dataset_label="positive",
        window_size=10,
        stride=1,
        converted_dir=tmp_path / "converted",
    )
    policy_windows, policy_meta = diagnostics.load_motion_windows(
        policy_windows_path,
        dataset_label="policy",
        window_size=10,
        stride=1,
    )

    assert raw_windows.shape == (3, 10, 192)
    assert raw_meta["converted_path"] is not None
    assert Path(raw_meta["converted_path"]).is_file()
    assert policy_windows.shape == (5, 10, 192)
    assert policy_meta["kind"] == "policy_windows"


def test_normalize_and_format_group_scores_orders_policy_between_positive_and_negative():
    diagnostics = _load_smp_diagnostics_module()

    positive = diagnostics.GroupScoreResult(
        label="positive",
        num_windows=3,
        raw_noise_mse=np.array([0.10, 0.12, 0.11], dtype=np.float32),
        per_timestep_raw_mse={
            22: np.array([0.10, 0.11, 0.09], dtype=np.float32),
            15: np.array([0.12, 0.13, 0.11], dtype=np.float32),
        },
        source_paths=["positive.npz"],
    )
    policy = diagnostics.GroupScoreResult(
        label="policy",
        num_windows=3,
        raw_noise_mse=np.array([0.20, 0.18, 0.22], dtype=np.float32),
        per_timestep_raw_mse={
            22: np.array([0.18, 0.17, 0.19], dtype=np.float32),
            15: np.array([0.22, 0.19, 0.25], dtype=np.float32),
        },
        source_paths=["policy.npz"],
    )
    negative = diagnostics.GroupScoreResult(
        label="negative",
        num_windows=3,
        raw_noise_mse=np.array([0.40, 0.45, 0.42], dtype=np.float32),
        per_timestep_raw_mse={
            22: np.array([0.39, 0.41, 0.40], dtype=np.float32),
            15: np.array([0.44, 0.49, 0.43], dtype=np.float32),
        },
        source_paths=["negative.npz"],
    )

    summaries = diagnostics.normalize_and_summarize_group_scores(
        [positive, policy, negative],
        reward_scale=1.0,
    )
    table = diagnostics.format_group_summary_table(summaries)

    assert summaries["positive"].reward_mean > summaries["policy"].reward_mean > summaries["negative"].reward_mean
    assert summaries["positive"].raw_noise_mse_mean < summaries["policy"].raw_noise_mse_mean < summaries["negative"].raw_noise_mse_mean
    assert "positive" in table
    assert "policy" in table
    assert "negative" in table


def test_build_window_collector_from_observation_infers_feature_dim():
    diagnostics = _load_smp_diagnostics_module()

    collector = diagnostics.build_window_collector_from_observation(
        {"smp_motion_window": torch.zeros(2, 30, dtype=torch.float32)},
        obs_group="smp_motion_window",
        window_size=10,
    )

    assert collector.obs_group == "smp_motion_window"
    assert collector.window_size == 10
    assert collector.feature_dim == 3
