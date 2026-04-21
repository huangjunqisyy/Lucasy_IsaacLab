import importlib.util
import json
import sys
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


def _load_compare_module():
    return _load_module(
        ("scripts", "imitation_learning", "smp", "compare_smp_policy_checkpoints.py"),
        "isaaclab_smp_checkpoint_compare_unit",
    )


def test_compare_policy_checkpoints_collects_rollouts_and_writes_comparison(tmp_path, monkeypatch):
    module = _load_compare_module()
    prior_checkpoint = tmp_path / "prior.pt"
    standing_checkpoint = tmp_path / "standing.pt"
    policy_checkpoint = tmp_path / "policy.pt"
    reference_a = tmp_path / "walk_a.npz"
    reference_b = tmp_path / "walk_b.npz"
    manifest_path = tmp_path / "walk_manifest.json"
    output_dir = tmp_path / "comparison"

    for path in (prior_checkpoint, standing_checkpoint, policy_checkpoint, reference_a, reference_b):
        path.write_bytes(b"placeholder")

    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"path": str(reference_a)},
                    {"path": str(reference_b)},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rollout_calls = []

    def fake_run_rollout_collection(*, checkpoint_path, output_path, task, rollout_steps, device, num_envs, headless, seed, agent):
        rollout_calls.append(
            {
                "checkpoint_path": str(checkpoint_path),
                "output_path": str(output_path),
                "task": task,
                "rollout_steps": rollout_steps,
                "device": device,
                "num_envs": num_envs,
                "headless": headless,
                "seed": seed,
                "agent": agent,
            }
        )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            windows=np.zeros((4, 10, 192), dtype=np.float32),
            window_size=np.array([10], dtype=np.int64),
            feature_dim=np.array([192], dtype=np.int64),
        )
        return output_path

    evaluation_calls = []

    def fake_evaluate_smp_reward(**kwargs):
        evaluation_calls.append(kwargs)
        return {
            "checkpoint_path": str(kwargs["checkpoint_path"]),
            "timesteps_k": [22, 15, 8],
            "reward_scale": 1.0,
            "window_size": int(kwargs["window_size"]),
            "stride": int(kwargs["stride"]),
            "batch_size": int(kwargs["batch_size"]),
            "noise_draws": int(kwargs["noise_draws"]),
            "groups": {
                "positive": {
                    "label": "positive",
                    "num_windows": 20,
                    "raw_noise_mse_mean": 0.25,
                    "raw_noise_mse_std": 0.01,
                    "normalized_noise_mse_mean": 1.0,
                    "normalized_noise_mse_std": 0.05,
                    "reward_mean": 0.40,
                    "reward_std": 0.02,
                    "reward_p10": 0.37,
                    "reward_p50": 0.40,
                    "reward_p90": 0.43,
                    "per_timestep_raw_mse_mean": {"22": 0.20, "15": 0.24, "8": 0.31},
                    "source_paths": [str(reference_a), str(reference_b)],
                    "converted_paths": [],
                },
                "policy": {
                    "label": "policy",
                    "num_windows": 4,
                    "raw_noise_mse_mean": 0.30,
                    "raw_noise_mse_std": 0.01,
                    "normalized_noise_mse_mean": 1.2,
                    "normalized_noise_mse_std": 0.05,
                    "reward_mean": 0.34,
                    "reward_std": 0.01,
                    "reward_p10": 0.33,
                    "reward_p50": 0.34,
                    "reward_p90": 0.35,
                    "per_timestep_raw_mse_mean": {"22": 0.28, "15": 0.31, "8": 0.36},
                    "source_paths": [str(output_dir / "target_policy_windows.npz")],
                    "converted_paths": [],
                },
                "negative": {
                    "label": "negative",
                    "num_windows": 4,
                    "raw_noise_mse_mean": 0.45,
                    "raw_noise_mse_std": 0.02,
                    "normalized_noise_mse_mean": 1.8,
                    "normalized_noise_mse_std": 0.06,
                    "reward_mean": 0.22,
                    "reward_std": 0.01,
                    "reward_p10": 0.20,
                    "reward_p50": 0.22,
                    "reward_p90": 0.24,
                    "per_timestep_raw_mse_mean": {"22": 0.40, "15": 0.46, "8": 0.52},
                    "source_paths": [str(output_dir / "standing_policy_windows.npz")],
                    "converted_paths": [],
                },
            },
        }

    monkeypatch.setattr(module, "run_rollout_collection", fake_run_rollout_collection)
    monkeypatch.setattr(module, "evaluate_smp_reward", fake_evaluate_smp_reward)

    summary = module.compare_policy_checkpoints(
        prior_checkpoint=prior_checkpoint,
        standing_checkpoint=standing_checkpoint,
        policy_checkpoint=policy_checkpoint,
        reference_paths=[manifest_path],
        task="Unitree-G1-Flat-SMP-Play-v0",
        output_dir=output_dir,
        window_size=10,
        rollout_steps=256,
        stride=2,
        batch_size=16,
        noise_draws=3,
        device="cuda:0",
        num_envs=32,
        headless=True,
        seed=123,
        agent="rsl_rl_cfg_entry_point",
        style_name="walk",
    )

    assert [call["checkpoint_path"] for call in rollout_calls] == [str(standing_checkpoint), str(policy_checkpoint)]
    assert all(call["task"] == "Unitree-G1-Flat-SMP-Play-v0" for call in rollout_calls)
    assert all(call["rollout_steps"] == 256 for call in rollout_calls)
    assert evaluation_calls[0]["positive_paths"] == [reference_a, reference_b]
    assert evaluation_calls[0]["negative_paths"] == [output_dir / "standing_policy_windows.npz"]
    assert evaluation_calls[0]["policy_paths"] == [output_dir / "target_policy_windows.npz"]
    assert evaluation_calls[0]["style_name"] == "walk"

    comparison = summary["comparison"]
    assert comparison["standing_raw_mse_over_walk"] == 1.8
    assert comparison["target_raw_mse_over_walk"] == 1.2
    assert comparison["standing_reward_gap_vs_walk"] == -0.18
    assert comparison["target_reward_gap_vs_walk"] == -0.06
    assert comparison["target_vs_standing_reward_gap"] == 0.12

    output_json = output_dir / "comparison_report.json"
    assert output_json.exists()
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["comparison"]["target_vs_standing_reward_gap"] == 0.12

    report = module.format_comparison_report(summary)
    assert "standing_raw_mse_over_walk" in report
    assert "target_vs_standing_reward_gap" in report


def test_run_rollout_collection_uses_unitree_play_collect_when_repo_root_is_provided(tmp_path, monkeypatch):
    module = _load_compare_module()
    unitree_root = tmp_path / "lucasy_unitree_rl_lab"
    play_collect = unitree_root / "scripts" / "rsl_rl" / "play_collect.py"
    play_collect.parent.mkdir(parents=True, exist_ok=True)
    play_collect.write_text("# stub\n", encoding="utf-8")

    captured = {}

    def fake_run_command(command, *, cwd=None):
        captured["command"] = list(command)
        captured["cwd"] = None if cwd is None else str(cwd)

    monkeypatch.setattr(module, "_run_command", fake_run_command)

    output_path = module.run_rollout_collection(
        checkpoint_path=tmp_path / "policy.pt",
        output_path=tmp_path / "windows.npz",
        task="Isaac-SMP-Velocity-Flat-G1-v0",
        rollout_steps=123,
        device="cuda:0",
        num_envs=16,
        headless=True,
        seed=7,
        agent="rsl_rl_cfg_entry_point",
        rollout_repo_root=unitree_root,
    )

    assert output_path == (tmp_path / "windows.npz").resolve()
    assert captured["cwd"] == str(unitree_root)
    assert captured["command"][1] == str(play_collect)
    assert "--output" in captured["command"]
    assert "--steps" in captured["command"]
    assert "--headless" in captured["command"]
