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


def _load_preview_module():
    module_path = Path(__file__).resolve().parents[1] / "isaaclab_rl" / "rsl_rl" / "smp_preview.py"
    spec = importlib.util.spec_from_file_location("isaaclab_smp_preview_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _install_fake_runner(monkeypatch, runner_cls):
    fake_package = types.ModuleType("rsl_rl")
    fake_package.__path__ = []
    fake_runners = types.ModuleType("rsl_rl.runners")
    fake_runners.SMPOnPolicyRunner = runner_cls
    fake_package.runners = fake_runners
    monkeypatch.setitem(sys.modules, "rsl_rl", fake_package)
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", fake_runners)


class _CfgNode:
    def __init__(self, **kwargs):
        self._values = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        output = {}
        for key, value in self._values.items():
            output[key] = value.to_dict() if hasattr(value, "to_dict") else value
        return output


def test_build_smp_preview_runtime_converts_config_objects_for_runner_loading(monkeypatch):
    preview_module = _load_preview_module()
    recorded = {}

    class _FakeRunner:
        def _load_prior_model(self):
            recorded["smp_prior_cfg"] = self.smp_prior_cfg
            recorded["style_cfg"] = self.style_cfg
            recorded["gsi_cfg"] = self.gsi_cfg
            self.smp_checkpoint_style_cfg = {"style_to_id": {"walk": 1}}
            return "fake-prior"

        def _resolve_style_program_from_cfg(self):
            return {"mode": "single_style", "guidance_scale": 1.25, "target_style_name": "walk"}

        def _build_gsi_sampler(self):
            return "fake-gsi-sampler"

    _install_fake_runner(monkeypatch, _FakeRunner)

    agent_cfg = types.SimpleNamespace(
        smp_prior=_CfgNode(
            checkpoint_path="/tmp/model.pt",
            feature_dim=131,
            window_size=10,
            num_diffusion_steps=50,
            style_cfg=_CfgNode(mode="single_style", target_style_name="walk"),
        ),
        gsi_cfg=_CfgNode(
            enabled=True,
            guidance_scale=None,
            fallback_to_default_reset=True,
            max_resample_attempts=4,
            asset_name="robot",
        ),
    )

    runtime = preview_module.build_smp_preview_runtime(agent_cfg=agent_cfg, device="cpu")

    assert recorded["smp_prior_cfg"] == {
        "checkpoint_path": "/tmp/model.pt",
        "feature_dim": 131,
        "window_size": 10,
        "num_diffusion_steps": 50,
        "style_cfg": {"mode": "single_style", "target_style_name": "walk"},
    }
    assert recorded["style_cfg"] == {"mode": "single_style", "target_style_name": "walk"}
    assert recorded["gsi_cfg"] == {
        "enabled": True,
        "guidance_scale": None,
        "fallback_to_default_reset": True,
        "max_resample_attempts": 4,
        "asset_name": "robot",
    }
    assert runtime.gsi_sampler == "fake-gsi-sampler"
    assert runtime.style_program["target_style_name"] == "walk"
    assert runtime.guidance_scale == pytest.approx(1.25)
    assert runtime.max_resample_attempts == 4
    assert runtime.fallback_to_default_reset is True
    assert runtime.asset_name == "robot"


def test_sample_and_apply_preview_reset_applies_supported_state(monkeypatch):
    preview_module = _load_preview_module()
    recorded = {}

    smp_reset_module = types.ModuleType("isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset")

    def _build_reference(env, env_ids=None, asset_name="robot"):
        recorded["reference_env_ids"] = env_ids.clone()
        recorded["asset_name"] = asset_name
        return "reference-state"

    def _apply_state(env, env_ids=None, state=None, asset_name="robot"):
        recorded["applied_env_ids"] = env_ids.clone()
        recorded["applied_state"] = state
        recorded["applied_asset_name"] = asset_name

    smp_reset_module.build_smp_reset_reference = _build_reference
    smp_reset_module.apply_smp_reset_state = _apply_state
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset", smp_reset_module)

    sampler_calls = []

    runtime = preview_module.SMPPreviewRuntime(
        gsi_sampler=types.SimpleNamespace(
            sample_reset_state=lambda batch_size, reference_state, style_program, guidance_scale: sampler_calls.append(
                {
                    "batch_size": batch_size,
                    "reference_state": reference_state,
                    "style_program": style_program,
                    "guidance_scale": guidance_scale,
                }
            )
            or types.SimpleNamespace(
                state="generated-state",
                supports_reset_state=True,
                reconstruction_mse=0.0,
                unrecoverable_feature_blocks=(),
            )
        ),
        style_program={"mode": "single_style", "target_style_name": "walk"},
        guidance_scale=1.75,
        max_resample_attempts=3,
        fallback_to_default_reset=True,
        asset_name="robot",
    )

    reset_result = preview_module.sample_and_apply_preview_reset(
        env=object(),
        runtime=runtime,
        env_ids=torch.tensor([2], dtype=torch.long),
    )

    assert sampler_calls == [
        {
            "batch_size": 1,
            "reference_state": "reference-state",
            "style_program": {"mode": "single_style", "target_style_name": "walk"},
            "guidance_scale": 1.75,
        }
    ]
    assert torch.equal(recorded["reference_env_ids"], torch.tensor([2], dtype=torch.long))
    assert torch.equal(recorded["applied_env_ids"], torch.tensor([2], dtype=torch.long))
    assert recorded["applied_state"] == "generated-state"
    assert reset_result.applied is True
    assert reset_result.fallback_to_default is False
    assert reset_result.sample_attempts == 1
    assert reset_result.decode_result.reconstruction_mse == pytest.approx(0.0)


def test_sample_and_apply_preview_reset_keeps_default_reset_when_all_samples_fail(monkeypatch):
    preview_module = _load_preview_module()
    recorded = {"apply_calls": 0}

    smp_reset_module = types.ModuleType("isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset")
    smp_reset_module.build_smp_reset_reference = lambda env, env_ids=None, asset_name="robot": "reference-state"
    smp_reset_module.apply_smp_reset_state = (
        lambda env, env_ids=None, state=None, asset_name="robot": recorded.__setitem__("apply_calls", recorded["apply_calls"] + 1)
    )
    monkeypatch.setitem(sys.modules, "isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset", smp_reset_module)

    sample_attempts = {"count": 0}

    def _sample_reset_state(batch_size, reference_state, style_program, guidance_scale):
        sample_attempts["count"] += 1
        return types.SimpleNamespace(
            state="unsupported-state",
            supports_reset_state=False,
            reconstruction_mse=0.5,
            unrecoverable_feature_blocks=("ee_pos_b",),
        )

    runtime = preview_module.SMPPreviewRuntime(
        gsi_sampler=types.SimpleNamespace(sample_reset_state=_sample_reset_state),
        style_program={"mode": "single_style", "target_style_name": "walk"},
        guidance_scale=1.0,
        max_resample_attempts=2,
        fallback_to_default_reset=True,
        asset_name="robot",
    )

    reset_result = preview_module.sample_and_apply_preview_reset(
        env=object(),
        runtime=runtime,
        env_ids=torch.tensor([0], dtype=torch.long),
    )

    assert sample_attempts["count"] == 2
    assert recorded["apply_calls"] == 0
    assert reset_result.applied is False
    assert reset_result.fallback_to_default is True
    assert reset_result.sample_attempts == 2
    assert reset_result.decode_result.reconstruction_mse == pytest.approx(0.5)
