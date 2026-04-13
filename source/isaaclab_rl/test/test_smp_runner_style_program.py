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
        def log(self, locs, width=80, pad=35):
            return None

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


class _FakeCircularBuffer:
    def __init__(self, data: torch.Tensor, pointer: int, num_pushes: torch.Tensor):
        self._buffer = data.clone()
        self._pointer = int(pointer)
        self._num_pushes = num_pushes.clone()
        self.max_length = data.shape[0]
        self.batch_size = data.shape[1]
        self.device = data.device

    @property
    def buffer(self) -> torch.Tensor:
        buf = self._buffer.clone()
        buf = torch.roll(buf, shifts=self.max_length - self._pointer - 1, dims=0)
        return torch.transpose(buf, dim0=0, dim1=1)


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


def test_smp_reward_obs_prefers_terminal_observation_for_done_envs(monkeypatch):
    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )
    runner = object.__new__(runner_module.SMPOnPolicyRunner)
    runner.smp_obs_group = "smp_motion_window"

    reward_obs = runner._build_smp_reward_obs(
        {
            "smp_motion_window": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "policy": torch.tensor([[10.0], [20.0]]),
        },
        torch.tensor([0, 1], dtype=torch.long),
        {
            "terminal_observation": {
                "smp_motion_window": torch.tensor([[30.0, 40.0]]),
            }
        },
    )

    assert torch.allclose(
        reward_obs["smp_motion_window"],
        torch.tensor([[1.0, 2.0], [30.0, 40.0]]),
    )


def test_gsi_reset_rebuilds_history_backed_observations_for_reset_envs(monkeypatch):
    recorded = {}

    smp_reset_module = types.ModuleType("isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset")

    def _build_reference(env, env_ids, asset_name="robot"):
        recorded["reference_env_ids"] = env_ids.clone()
        return "reference-state"

    def _apply_state(env, env_ids, state, asset_name="robot"):
        recorded["applied_env_ids"] = env_ids.clone()
        recorded["applied_state"] = state
        env.unwrapped.policy_frame[env_ids] = torch.tensor([[91.0]], dtype=torch.float32)
        env.unwrapped.smp_frame[env_ids] = torch.tensor([[81.0, 82.0]], dtype=torch.float32)

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
    runner.style_program = {"mode": "single_style", "guidance_scale": 1.0, "target_style_id": 1, "target_style_name": "walk"}
    runner.gsi_cfg = {
        "enabled": True,
        "sample_on_reset": True,
        "guidance_scale": None,
        "fallback_to_default_reset": True,
        "max_resample_attempts": 1,
        "asset_name": "robot",
    }
    runner.gsi_sampler = types.SimpleNamespace(
        sample_reset_state=lambda batch_size, reference_state, style_program, guidance_scale: types.SimpleNamespace(
            state="generated-state",
            supports_reset_state=True,
        )
    )

    policy_buffer = _FakeCircularBuffer(
        data=torch.tensor(
            [
                [[11.0], [20.0]],
                [[12.0], [20.0]],
            ],
            dtype=torch.float32,
        ),
        pointer=1,
        num_pushes=torch.tensor([2, 1], dtype=torch.long),
    )
    smp_buffer = _FakeCircularBuffer(
        data=torch.tensor(
            [
                [[101.0, 102.0], [30.0, 40.0]],
                [[103.0, 104.0], [30.0, 40.0]],
            ],
            dtype=torch.float32,
        ),
        pointer=1,
        num_pushes=torch.tensor([2, 1], dtype=torch.long),
    )

    fake_obs_manager = types.SimpleNamespace(
        _env=None,
        _group_obs_term_names={"policy": ["policy_frame"], "smp_motion_window": ["motion_frame"]},
        _group_obs_term_cfgs={
            "policy": [
                types.SimpleNamespace(
                    func=lambda env, **_: env.policy_frame,
                    params={},
                    modifiers=None,
                    noise=None,
                    clip=None,
                    scale=None,
                    history_length=2,
                    flatten_history_dim=True,
                )
            ],
            "smp_motion_window": [
                types.SimpleNamespace(
                    func=lambda env, **_: env.smp_frame,
                    params={},
                    modifiers=None,
                    noise=None,
                    clip=None,
                    scale=None,
                    history_length=2,
                    flatten_history_dim=True,
                )
            ],
        },
        _group_obs_term_history_buffer={
            "policy": {"policy_frame": policy_buffer},
            "smp_motion_window": {"motion_frame": smp_buffer},
        },
        _group_obs_concatenate={"policy": True, "smp_motion_window": True},
        _group_obs_concatenate_dim={"policy": -1, "smp_motion_window": -1},
    )
    fake_unwrapped = types.SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        policy_frame=torch.tensor([[12.0], [20.0]], dtype=torch.float32),
        smp_frame=torch.tensor([[103.0, 104.0], [30.0, 40.0]], dtype=torch.float32),
        observation_manager=fake_obs_manager,
    )
    fake_obs_manager._env = fake_unwrapped
    runner.env = types.SimpleNamespace(
        num_envs=2,
        get_observations=lambda: (_ for _ in ()).throw(AssertionError("history-backed refresh should not use get_observations")),
        unwrapped=fake_unwrapped,
    )

    obs, diag = runner._maybe_apply_gsi_reset(
        {
            "policy": policy_buffer.buffer.reshape(2, -1),
            "smp_motion_window": smp_buffer.buffer.reshape(2, -1),
        },
        torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.equal(recorded["reference_env_ids"], torch.tensor([1], dtype=torch.long))
    assert torch.equal(recorded["applied_env_ids"], torch.tensor([1], dtype=torch.long))
    assert recorded["applied_state"] == "generated-state"
    assert torch.allclose(obs["policy"], torch.tensor([[11.0, 12.0], [91.0, 91.0]]))
    assert torch.allclose(
        obs["smp_motion_window"],
        torch.tensor([[101.0, 102.0, 103.0, 104.0], [81.0, 82.0, 81.0, 82.0]]),
    )
    assert diag["reset_accept_rate"] == pytest.approx(1.0)


def test_smp_log_reports_reward_breakdown(monkeypatch, capsys):
    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )

    class _FakeWriter:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, float(value), int(step)))

    runner = object.__new__(runner_module.SMPOnPolicyRunner)
    runner.writer = _FakeWriter()
    runner.log_histograms_every = 20

    runner.log(
        {
            "it": 1,
            "smp_mean_reward": 0.36,
            "smp_noise_mse": 1.0,
            "smp_cfg_gap": 0.05,
            "gsi_reset_accept_rate": 1.0,
            "gsi_reset_resample_count": 0.0,
            "gsi_fallback_rate": 0.0,
            "smp_per_timestep_mse": {22: 1.1, 15: 0.9},
            "smp_eps": {},
            "smp_eps_hat": {},
            "style_program": {
                "mode": "single_style",
                "target_style_id": 1,
                "target_style_name": "walk",
            },
            "style_diag": {},
            "smp_task_reward_raw": 1.2,
            "smp_task_reward_scaled": 1.2,
            "smp_style_reward_raw": 0.36,
            "smp_style_reward_scaled": 0.00144,
            "smp_combined_reward": 1.20144,
        }
    )

    scalar_tags = {tag for tag, _, _ in runner.writer.scalars}

    assert "SMP/reward_terms/task_raw" in scalar_tags
    assert "SMP/reward_terms/task_scaled" in scalar_tags
    assert "SMP/reward_terms/style_raw" in scalar_tags
    assert "SMP/reward_terms/style_scaled" in scalar_tags
    assert "SMP/reward_terms/combined" in scalar_tags

    captured = capsys.readouterr()
    assert "SMP reward diagnostics:" in captured.out
    assert "Mean task reward raw:" in captured.out
    assert "Mean task reward scaled:" in captured.out
    assert "Mean SMP reward raw:" in captured.out
    assert "Mean SMP reward scaled:" in captured.out
    assert "Mean combined reward:" in captured.out


def test_smp_log_reports_target_vs_uncond_reward_terms(monkeypatch):
    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )

    class _FakeWriter:
        def __init__(self):
            self.scalars = []

        def add_scalar(self, tag, value, step):
            self.scalars.append((tag, float(value), int(step)))

    runner = object.__new__(runner_module.SMPOnPolicyRunner)
    runner.writer = _FakeWriter()
    runner.log_histograms_every = 20

    runner.log(
        {
            "it": 1,
            "smp_mean_reward": 0.65,
            "smp_noise_mse": 0.25,
            "smp_cfg_gap": 0.05,
            "gsi_reset_accept_rate": 1.0,
            "gsi_reset_resample_count": 0.0,
            "gsi_fallback_rate": 0.0,
            "smp_per_timestep_mse": {22: 0.3, 15: 0.2},
            "smp_eps": {},
            "smp_eps_hat": {},
            "style_program": {
                "mode": "single_style",
                "target_style_id": 1,
                "target_style_name": "walk",
            },
            "style_diag": {},
            "smp_task_reward_raw": 1.2,
            "smp_task_reward_scaled": 1.2,
            "smp_style_reward_raw": 0.65,
            "smp_style_reward_scaled": 0.0026,
            "smp_combined_reward": 1.2026,
            "smp_reward_target": 0.78,
            "smp_reward_uncond": 0.37,
            "smp_reward_gap": 0.41,
            "smp_reward_final": 0.65,
        }
    )

    scalar_tags = {tag for tag, _, _ in runner.writer.scalars}

    assert "SMP/diff/reward_target" in scalar_tags
    assert "SMP/diff/reward_uncond" in scalar_tags
    assert "SMP/diff/reward_gap" in scalar_tags
    assert "SMP/diff/reward_final" in scalar_tags


def test_compute_smp_metrics_passes_uncond_prediction_in_target_vs_uncond_mode(monkeypatch):
    runner_module = _load_runner_module(
        monkeypatch,
        feature_masks={
            "shared_body": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "lower_body": torch.tensor([0.0, 1.0, 0.0, 0.0]),
            "upper_body": torch.tensor([0.0, 0.0, 1.0, 1.0]),
        },
    )

    class _FakeReward:
        timesteps_k = [22, 15, 8]
        reward_mode = "target_vs_uncond"

        def __init__(self):
            self.recorded = None

        def compute(self, eps, eps_hat, eps_hat_uncond=None):
            self.recorded = {
                "eps": eps,
                "eps_hat": eps_hat,
                "eps_hat_uncond": eps_hat_uncond,
            }
            return {
                "reward": torch.ones(2),
                "noise_mse": torch.full((2,), 0.5),
                "per_timestep_mse": {22: torch.full((2,), 0.2), 15: torch.full((2,), 0.3), 8: torch.full((2,), 0.4)},
            }

    runner = object.__new__(runner_module.SMPOnPolicyRunner)
    runner.device = torch.device("cpu")
    runner.smp_obs_group = "smp_motion_window"
    runner.smp_prior_cfg = {"feature_dim": 2, "window_size": 2}
    runner.smp_reward = _FakeReward()
    runner.smp_scheduler = types.SimpleNamespace(q_sample=lambda x0, t, eps_t: x0)
    runner._predict_prior_eps = lambda xt, t: (
        {
            "policy": torch.full_like(xt, 3.0),
            "target": torch.full_like(xt, 2.0),
            "uncond": torch.full_like(xt, 4.0),
        },
        {"cond_uncond_gap": 0.25},
    )

    metrics = runner._compute_smp_metrics({"smp_motion_window": torch.zeros(2, 4)})

    assert metrics["reward"].shape == (2,)
    assert runner.smp_reward.recorded is not None
    assert runner.smp_reward.recorded["eps_hat_uncond"] is not None
    assert set(runner.smp_reward.recorded["eps_hat_uncond"].keys()) == {22, 15, 8}
