# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def _load_module(relative_parts: tuple[str, ...], module_name: str):
    for parent in Path(__file__).resolve().parents:
        module_path = parent.joinpath(*relative_parts)
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Could not find module: {'/'.join(relative_parts)}")


def _load_evaluation_module():
    return _load_module(
        ("scripts", "imitation_learning", "smp", "evaluate_smp_reward.py"),
        "isaaclab_smp_reward_evaluation_unit",
    )


def test_evaluate_smp_reward_compares_positive_policy_and_negative_groups(tmp_path, monkeypatch):
    module = _load_evaluation_module()
    positive_path = tmp_path / "positive_walk.npz"
    negative_path = tmp_path / "negative_jump.npz"
    policy_path = tmp_path / "policy_windows.npz"
    output_json = tmp_path / "summary.json"
    converted_dir = tmp_path / "converted"

    for path in (positive_path, negative_path, policy_path):
        path.write_bytes(b"placeholder")

    load_calls = []

    def fake_load_motion_windows(input_path, *, dataset_label, window_size, stride, converted_dir=None):
        load_calls.append(
            {
                "input_path": str(input_path),
                "dataset_label": dataset_label,
                "window_size": window_size,
                "stride": stride,
                "converted_dir": None if converted_dir is None else str(converted_dir),
            }
        )
        num_windows = {"positive": 2, "policy": 3, "negative": 4}[dataset_label]
        converted_path = None
        if converted_dir is not None:
            converted_path = Path(converted_dir) / f"{dataset_label}_{Path(input_path).stem}.npz"
            converted_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(converted_path, frames=np.zeros((12, 192), dtype=np.float32))
        return np.full((num_windows, 10, 192), fill_value=float(num_windows), dtype=np.float32), {
            "kind": "policy_windows" if dataset_label == "policy" else "frames_dataset",
            "source_path": str(input_path),
            "converted_path": None if converted_path is None else str(converted_path),
        }

    prior = types.SimpleNamespace(
        reward_scale=1.25,
        timesteps_k=[22, 15, 8],
        style_id=None,
        style_name=None,
    )

    def fake_score_motion_windows(windows, *, prior, batch_size, noise_draws, device, seed, label, source_paths):
        mse = {"positive": 0.10, "policy": 0.20, "negative": 0.40}[label]
        return module.GroupScoreResult(
            label=label,
            num_windows=int(windows.shape[0]),
            raw_noise_mse=np.full((windows.shape[0],), mse, dtype=np.float32),
            per_timestep_raw_mse={
                22: np.full((windows.shape[0],), mse + 0.01, dtype=np.float32),
                15: np.full((windows.shape[0],), mse + 0.02, dtype=np.float32),
                8: np.full((windows.shape[0],), mse + 0.03, dtype=np.float32),
            },
            source_paths=list(source_paths),
        )

    monkeypatch.setattr(module, "load_motion_windows", fake_load_motion_windows)
    monkeypatch.setattr(module, "load_prior_bundle", lambda *args, **kwargs: prior)
    monkeypatch.setattr(module, "score_motion_windows", fake_score_motion_windows)

    summary = module.evaluate_smp_reward(
        checkpoint_path=tmp_path / "prior.pt",
        positive_paths=[positive_path],
        negative_paths=[negative_path],
        policy_paths=[policy_path],
        window_size=10,
        stride=1,
        batch_size=8,
        noise_draws=2,
        device="cpu",
        converted_dir=converted_dir,
        output_json=output_json,
    )
    report = module.format_evaluation_report(summary)

    assert [call["dataset_label"] for call in load_calls] == ["positive", "negative", "policy"]
    assert load_calls[0]["converted_dir"] == str(converted_dir / "positive")
    assert load_calls[1]["converted_dir"] == str(converted_dir / "negative")
    assert load_calls[2]["converted_dir"] is None
    assert summary["groups"]["positive"]["num_windows"] == 2
    assert summary["groups"]["policy"]["num_windows"] == 3
    assert summary["groups"]["negative"]["num_windows"] == 4
    assert summary["groups"]["positive"]["reward_mean"] > summary["groups"]["policy"]["reward_mean"]
    assert summary["groups"]["policy"]["reward_mean"] > summary["groups"]["negative"]["reward_mean"]
    assert json.loads(output_json.read_text(encoding="utf-8"))["groups"]["policy"]["num_windows"] == 3
    assert "checkpoint_path:" in report
    assert "positive" in report
    assert "policy" in report
    assert "negative" in report
