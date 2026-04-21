#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PATH_BOOTSTRAP = _load_module(
    "isaaclab_smp_path_bootstrap_compare_checkpoints",
    Path(__file__).resolve().with_name("path_bootstrap.py"),
)
_PATH_BOOTSTRAP.prepend_local_source_paths(__file__, package_names=("isaaclab", "isaaclab_rl", "isaaclab_tasks"))
_REPO_ROOT = _PATH_BOOTSTRAP.find_repo_root(__file__)
_EVALUATION_MODULE = _load_module(
    "isaaclab_smp_reward_compare_checkpoint_eval",
    Path(__file__).resolve().with_name("evaluate_smp_reward.py"),
)

evaluate_smp_reward = _EVALUATION_MODULE.evaluate_smp_reward
format_evaluation_report = _EVALUATION_MODULE.format_evaluation_report


def _resolve_dataset_manifest(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        entries = payload.get("datasets", payload.get("motions", payload.get("files", [])))
    elif isinstance(payload, list):
        entries = payload
    else:
        raise ValueError(f"Unsupported manifest format in {path}")

    resolved_paths = []
    for entry in entries:
        if isinstance(entry, str):
            candidate = Path(entry)
        elif isinstance(entry, dict) and "path" in entry:
            candidate = Path(entry["path"])
        else:
            raise ValueError(f"Unsupported manifest entry in {path}: {entry!r}")
        if not candidate.is_absolute():
            candidate = path.parent / candidate
        resolved_paths.append(candidate.expanduser().resolve())

    if not resolved_paths:
        raise ValueError(f"Manifest contains no dataset paths: {path}")
    return resolved_paths


def expand_dataset_paths(paths: Sequence[str | Path]) -> list[Path]:
    expanded = []
    for path_like in paths:
        path = Path(path_like).expanduser().resolve()
        if path.suffix.lower() == ".json":
            expanded.extend(_resolve_dataset_manifest(path))
        else:
            expanded.append(path)
    if not expanded:
        raise ValueError("reference_paths must be non-empty")
    return expanded


def _run_command(command: Sequence[str], *, cwd: str | Path | None = None) -> None:
    subprocess.run(list(command), cwd=None if cwd is None else str(cwd), check=True)


def _resolve_rollout_entry(
    rollout_repo_root: str | Path | None = None,
) -> tuple[Path, Path, str]:
    candidate_roots = []
    if rollout_repo_root is not None:
        candidate_roots.append(Path(rollout_repo_root).expanduser().resolve())

    for env_name in ("LUCASY_UNITREE_RL_LAB_ROOT", "UNITREE_RL_LAB_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            candidate_roots.append(Path(env_value).expanduser().resolve())

    cwd = Path.cwd().resolve()
    candidate_roots.extend([cwd, *cwd.parents])
    candidate_roots.append((_REPO_ROOT.parent / "lucasy_unitree_rl_lab").resolve())

    seen_roots = set()
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)

        play_collect = resolved_root / "scripts" / "rsl_rl" / "play_collect.py"
        if play_collect.exists():
            return resolved_root, play_collect, "unitree_play_collect"

    play_script = _REPO_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl" / "play.py"
    return _REPO_ROOT, play_script, "isaaclab_play"


def run_rollout_collection(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    task: str,
    rollout_steps: int,
    device: str,
    num_envs: int | None,
    headless: bool,
    seed: int | None,
    agent: str,
    rollout_repo_root: str | Path | None = None,
) -> Path:
    rollout_root, rollout_script, rollout_mode = _resolve_rollout_entry(rollout_repo_root)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(rollout_script),
        "--task",
        str(task),
        "--agent",
        str(agent),
        "--checkpoint",
        str(Path(checkpoint_path).expanduser().resolve()),
        "--device",
        str(device),
    ]
    if rollout_mode == "unitree_play_collect":
        command.extend(["--output", str(output_path), "--steps", str(int(rollout_steps))])
    else:
        command.extend(
            [
                "--dump_smp_windows",
                "--dump_smp_windows_output",
                str(output_path),
                "--dump_smp_windows_steps",
                str(int(rollout_steps)),
            ]
        )
    if num_envs is not None:
        command.extend(["--num_envs", str(int(num_envs))])
    if headless:
        command.append("--headless")
    if seed is not None:
        command.extend(["--seed", str(int(seed))])

    _run_command(command, cwd=rollout_root)
    return output_path


def _safe_ratio(numerator: float, denominator: float) -> float:
    numerator = float(numerator)
    denominator = float(denominator)
    if abs(denominator) < 1.0e-8:
        return 1.0 if abs(numerator) < 1.0e-8 else float("inf")
    return numerator / denominator


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _build_comparison_metrics(evaluation_summary: dict[str, Any]) -> dict[str, float]:
    groups = evaluation_summary["groups"]
    positive = groups["positive"]
    negative = groups["negative"]
    policy = groups["policy"]
    return {
        "standing_raw_mse_over_walk": _rounded(
            _safe_ratio(negative["raw_noise_mse_mean"], positive["raw_noise_mse_mean"])
        ),
        "target_raw_mse_over_walk": _rounded(
            _safe_ratio(policy["raw_noise_mse_mean"], positive["raw_noise_mse_mean"])
        ),
        "standing_reward_gap_vs_walk": _rounded(negative["reward_mean"] - positive["reward_mean"]),
        "target_reward_gap_vs_walk": _rounded(policy["reward_mean"] - positive["reward_mean"]),
        "target_vs_standing_reward_gap": _rounded(policy["reward_mean"] - negative["reward_mean"]),
    }


def compare_policy_checkpoints(
    *,
    prior_checkpoint: str | Path,
    standing_checkpoint: str | Path,
    policy_checkpoint: str | Path,
    reference_paths: Sequence[str | Path],
    task: str,
    output_dir: str | Path,
    window_size: int,
    rollout_steps: int,
    stride: int = 1,
    batch_size: int = 128,
    noise_draws: int = 4,
    device: str = "cpu",
    num_envs: int | None = None,
    headless: bool = True,
    seed: int | None = None,
    agent: str = "rsl_rl_cfg_entry_point",
    rollout_repo_root: str | Path | None = None,
    style_name: str | None = None,
    style_id: int | None = None,
    timesteps_k: Sequence[int] | None = None,
    reward_scale: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    standing_windows_path = output_dir / "standing_policy_windows.npz"
    target_windows_path = output_dir / "target_policy_windows.npz"
    expanded_reference_paths = expand_dataset_paths(reference_paths)
    rollout_kwargs = {}
    if rollout_repo_root is not None:
        rollout_kwargs["rollout_repo_root"] = rollout_repo_root

    run_rollout_collection(
        checkpoint_path=standing_checkpoint,
        output_path=standing_windows_path,
        task=task,
        rollout_steps=rollout_steps,
        device=device,
        num_envs=num_envs,
        headless=headless,
        seed=seed,
        agent=agent,
        **rollout_kwargs,
    )
    run_rollout_collection(
        checkpoint_path=policy_checkpoint,
        output_path=target_windows_path,
        task=task,
        rollout_steps=rollout_steps,
        device=device,
        num_envs=num_envs,
        headless=headless,
        seed=seed,
        agent=agent,
        **rollout_kwargs,
    )

    evaluation_summary = evaluate_smp_reward(
        checkpoint_path=prior_checkpoint,
        positive_paths=expanded_reference_paths,
        negative_paths=[standing_windows_path],
        policy_paths=[target_windows_path],
        window_size=int(window_size),
        stride=int(stride),
        batch_size=int(batch_size),
        noise_draws=int(noise_draws),
        device=device,
        converted_dir=output_dir / "converted",
        output_json=output_dir / "evaluation_summary.json",
        style_name=style_name,
        style_id=style_id,
        timesteps_k=None if timesteps_k is None else [int(timestep) for timestep in timesteps_k],
        reward_scale=reward_scale,
    )

    payload = {
        "prior_checkpoint": str(Path(prior_checkpoint).expanduser().resolve()),
        "standing_checkpoint": str(Path(standing_checkpoint).expanduser().resolve()),
        "policy_checkpoint": str(Path(policy_checkpoint).expanduser().resolve()),
        "task": str(task),
        "reference_paths": [str(path) for path in expanded_reference_paths],
        "rollout_repo_root": str(_resolve_rollout_entry(rollout_repo_root)[0]),
        "standing_windows_path": str(standing_windows_path),
        "target_windows_path": str(target_windows_path),
        "comparison": _build_comparison_metrics(evaluation_summary),
        "evaluation": evaluation_summary,
    }

    output_json = output_dir / "comparison_report.json"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def format_comparison_report(summary: dict[str, Any]) -> str:
    lines = [
        f"prior_checkpoint: {summary['prior_checkpoint']}",
        f"standing_checkpoint: {summary['standing_checkpoint']}",
        f"policy_checkpoint: {summary['policy_checkpoint']}",
        f"task: {summary['task']}",
        f"reference_paths: {summary['reference_paths']}",
        f"rollout_repo_root: {summary['rollout_repo_root']}",
        f"standing_windows_path: {summary['standing_windows_path']}",
        f"target_windows_path: {summary['target_windows_path']}",
        "",
        "[derived diagnostics]",
    ]
    for key, value in summary["comparison"].items():
        lines.append(f"{key}: {value:.6f}")
    lines.extend(["", "[evaluation]", format_evaluation_report(summary["evaluation"])])
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 standing policy / locomotion policy checkpoint 做 SMP 先验对比诊断。")
    parser.add_argument("--prior-checkpoint", required=True, help="SMP prior checkpoint，例如 model_latest.pt。")
    parser.add_argument("--standing-checkpoint", required=True, help="静止/保守策略 checkpoint。")
    parser.add_argument("--policy-checkpoint", required=True, help="目标 locomotion 策略 checkpoint。")
    parser.add_argument("--reference", nargs="+", required=True, help="walk 参考 npz，或 manifest json。")
    parser.add_argument("--task", required=True, help="IsaacLab 任务名，供 play.py rollout 使用。")
    parser.add_argument("--output-dir", required=True, help="输出目录。")
    parser.add_argument("--window-size", type=int, required=True, help="SMP 窗口长度。")
    parser.add_argument("--rollout-steps", type=int, default=2000, help="每个 checkpoint 收集的 rollout step 数。")
    parser.add_argument("--stride", type=int, default=1, help="参考数据转滑窗时的 stride。")
    parser.add_argument("--batch-size", type=int, default=128, help="离线打分 batch size。")
    parser.add_argument("--noise-draws", type=int, default=4, help="每个 timestep 的噪声重采样次数。")
    parser.add_argument("--device", default="cpu", help="rollout 与评估设备，例如 cuda:0。")
    parser.add_argument("--num-envs", type=int, default=None, help="rollout 时并行环境数。")
    parser.add_argument("--seed", type=int, default=None, help="rollout 随机种子。")
    parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="play.py 的 agent entry point。")
    parser.add_argument("--rollout-repo-root", default=None, help="可选：显式指定 rollout 使用的项目根目录。")
    parser.add_argument("--show-sim", action="store_true", default=False, help="默认 headless；加上本参数则显示仿真窗口。")
    parser.add_argument("--style-name", default=None, help="多风格 prior 的 style 名称。")
    parser.add_argument("--style-id", type=int, default=None, help="多风格 prior 的 style id。")
    parser.add_argument("--timesteps-k", type=int, nargs="+", default=None, help="覆盖 prior checkpoint 里的奖励 timestep 集合。")
    parser.add_argument("--reward-scale", type=float, default=None, help="覆盖 prior checkpoint 里的 reward scale。")
    return parser


def main() -> None:
    args = _build_argparser().parse_args()
    summary = compare_policy_checkpoints(
        prior_checkpoint=args.prior_checkpoint,
        standing_checkpoint=args.standing_checkpoint,
        policy_checkpoint=args.policy_checkpoint,
        reference_paths=args.reference,
        task=args.task,
        output_dir=args.output_dir,
        window_size=args.window_size,
        rollout_steps=args.rollout_steps,
        stride=args.stride,
        batch_size=args.batch_size,
        noise_draws=args.noise_draws,
        device=args.device,
        num_envs=args.num_envs,
        headless=not args.show_sim,
        seed=args.seed,
        agent=args.agent,
        rollout_repo_root=args.rollout_repo_root,
        style_name=args.style_name,
        style_id=args.style_id,
        timesteps_k=args.timesteps_k,
        reward_scale=args.reward_scale,
    )
    print(format_comparison_report(summary))
    print(f"\ncomparison_json: {Path(args.output_dir).expanduser().resolve() / 'comparison_report.json'}")
    print(f"evaluation_json: {Path(args.output_dir).expanduser().resolve() / 'evaluation_summary.json'}")


if __name__ == "__main__":
    main()
