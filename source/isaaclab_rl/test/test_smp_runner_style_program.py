# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_runner_module(monkeypatch, feature_masks: dict[str, torch.Tensor]):
    """通过伪造依赖模块加载 SMP runner，避免单元测试依赖完整 rsl_rl 安装。"""
    fake_package = types.ModuleType("rsl_rl")
    fake_package.__file__ = "/tmp/fake_rsl_rl/__init__.py"
    fake_package.__path__ = []

    fake_diffusion = types.ModuleType("rsl_rl.diffusion")
    fake_diffusion.NULL_STYLE_ID = -1
    fake_diffusion.DiffusionScheduler = object
    fake_diffusion.MotionEpsilonTransformer = object
    fake_diffusion.SMPDiffusionSampler = object
    fake_diffusion.SMPFeatureLayout = types.SimpleNamespace(from_feature_block_offsets=lambda offsets: offsets)
    fake_diffusion.SMPGSIDecoder = object
    fake_diffusion.SMPGSISampler = object
    fake_diffusion.SMPReward = object
    fake_diffusion.apply_classifier_free_guidance = lambda eps_uncond, eps_cond, guidance_scale: eps_cond
    fake_diffusion.build_g1_body_part_feature_masks = lambda **_: feature_masks
    fake_diffusion.compose_style_predictions_with_body_masks = lambda part_to_eps, masks: next(iter(part_to_eps.values()))
    fake_diffusion.log_smp_noise_metrics = lambda *args, **kwargs: None

    fake_runners = types.ModuleType("rsl_rl.runners")
    fake_runners.__path__ = []
    fake_on_policy_runner = types.ModuleType("rsl_rl.runners.on_policy_runner")

    class OnPolicyRunner:
        pass

    fake_on_policy_runner.OnPolicyRunner = OnPolicyRunner

    fake_utils = types.ModuleType("rsl_rl.utils")
    fake_utils.store_code_state = lambda *args, **kwargs: []

    monkeypatch.setitem(sys.modules, "rsl_rl", fake_package)
    monkeypatch.setitem(sys.modules, "rsl_rl.diffusion", fake_diffusion)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", fake_runners)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners.on_policy_runner", fake_on_policy_runner)
    monkeypatch.setitem(sys.modules, "rsl_rl.utils", fake_utils)

    for parent in Path(__file__).resolve().parents:
        module_path = parent / "rsl_rl" / "rsl_rl" / "runners" / "smp_on_policy_runner.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_runner_style_program_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find rsl_rl/rsl_rl/runners/smp_on_policy_runner.py")


def _make_runner(module, style_cfg, style_to_id):
    """构造只包含 style 解析所需字段的轻量 runner 实例。"""
    runner = object.__new__(module.SMPOnPolicyRunner)
    runner.smp_prior = types.SimpleNamespace(num_styles=len(style_to_id))
    runner.style_cfg = style_cfg
    runner.smp_checkpoint_style_cfg = {"style_to_id": style_to_id}
    return runner


def test_single_style_program_keeps_resolved_style_name(monkeypatch):
    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )
    runner = _make_runner(
        runner_module,
        style_cfg={"mode": "single_style", "target_style_name": "walk", "guidance_scale": 1.5},
        style_to_id={"dance": 0, "walk": 1},
    )

    style_program = runner._resolve_style_program_from_cfg()

    assert style_program == {
        "mode": "single_style",
        "guidance_scale": 1.5,
        "target_style_id": 1,
        "target_style_name": "walk",
    }


def test_body_mask_program_records_completed_part_style_names(monkeypatch):
    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )
    runner = _make_runner(
        runner_module,
        style_cfg={
            "mode": "body_mask",
            "guidance_scale": 1.0,
            "mask_name": "g1_upper_lower",
            "body_part_style_names": {"upper_body": "a", "lower_body": "c"},
            "joint_name_order": [],
            "ee_name_order": [],
            "key_body_name_order": [],
            "feature_block_offsets": {},
        },
        style_to_id={"a": 0, "b": 1, "c": 2},
    )

    style_program = runner._resolve_style_program_from_cfg()

    assert style_program["part_style_ids"] == {"shared_body": 2, "upper_body": 0, "lower_body": 2}
    assert style_program["part_style_names"] == {"shared_body": "c", "upper_body": "a", "lower_body": "c"}
    assert style_program["shared_body_defaulted"] is True
    assert style_program["mask_name"] == "g1_upper_lower"
    assert style_program["coverage"] == pytest.approx(1.0)


def test_gsi_reset_applies_sampled_state_and_refreshes_observations(monkeypatch):
    recorded = {}

    smp_reset_module = types.ModuleType("isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset")

    def _build_reference(env, env_ids, asset_name="robot"):
        recorded["reference_env_ids"] = env_ids.clone()
        return "reference-state"

    def _apply_state(env, env_ids, state, asset_name="robot"):
        recorded["applied_env_ids"] = env_ids.clone()
        recorded["applied_state"] = state

    smp_reset_module.build_smp_reset_reference = _build_reference
    smp_reset_module.apply_smp_reset_state = _apply_state
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset", smp_reset_module)

    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )
    runner = object.__new__(runner_module.SMPOnPolicyRunner)
    runner.device = torch.device("cpu")
    runner.env = types.SimpleNamespace(
        get_observations=lambda: {"policy": torch.tensor([[10.0], [20.0]])},
        unwrapped=types.SimpleNamespace(),
    )
    runner.style_program = {"mode": "single_style", "guidance_scale": 1.0, "target_style_id": 1, "target_style_name": "walk"}
    runner.gsi_cfg = {
        "enabled": True,
        "sample_on_reset": True,
        "guidance_scale": None,
        "fallback_to_default_reset": True,
        "max_resample_attempts": 2,
        "asset_name": "robot",
    }
    runner.gsi_sampler = types.SimpleNamespace(
        sample_reset_state=lambda batch_size, reference_state, style_program, guidance_scale: types.SimpleNamespace(
            state="generated-state",
            supports_reset_state=True,
            reconstruction_mse=0.0,
            unrecoverable_feature_blocks=("ee_pos_b",),
        )
    )

    obs, diag = runner._maybe_apply_gsi_reset(
        {"policy": torch.zeros(2, 1)},
        torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.equal(recorded["reference_env_ids"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(recorded["applied_env_ids"], torch.tensor([1], dtype=torch.long))
    assert recorded["applied_state"] == "generated-state"
    assert torch.allclose(obs["policy"], torch.tensor([[10.0], [20.0]]))
    assert diag["reset_accept_rate"] == pytest.approx(1.0)
    assert diag["reset_resample_count"] == pytest.approx(0.0)
    assert diag["fallback_rate"] == pytest.approx(0.0)
