# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PATH_BOOTSTRAP = _load_module(
    "isaaclab_smp_path_bootstrap_eval",
    Path(__file__).resolve().with_name("path_bootstrap.py"),
)
_PATH_BOOTSTRAP.prepend_local_source_paths(__file__, package_names=("isaaclab", "isaaclab_rl", "isaaclab_tasks"))
_REPO_ROOT = _PATH_BOOTSTRAP.find_repo_root(__file__)
_SMP_DIAGNOSTICS = _load_module(
    "isaaclab_smp_reward_eval_diagnostics",
    _REPO_ROOT / "source" / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "smp_diagnostics.py",
)
GroupScoreResult = _SMP_DIAGNOSTICS.GroupScoreResult
format_group_detail_lines = _SMP_DIAGNOSTICS.format_group_detail_lines
format_group_summary_table = _SMP_DIAGNOSTICS.format_group_summary_table
load_motion_windows = _SMP_DIAGNOSTICS.load_motion_windows
load_prior_bundle = _SMP_DIAGNOSTICS.load_prior_bundle
normalize_and_summarize_group_scores = _SMP_DIAGNOSTICS.normalize_and_summarize_group_scores
score_motion_windows = _SMP_DIAGNOSTICS.score_motion_windows

_REPORT_GROUP_ORDER = ("positive", "policy", "negative")


def _default_converted_dir(checkpoint_path: str | Path, output_json: str | Path | None) -> Path:
    if output_json is not None:
        return Path(output_json).resolve().parent / "converted"
    return Path(checkpoint_path).resolve().parent / "smp_reward_eval_converted"


def _score_group(
    *,
    label: str,
    input_paths: list[str | Path],
    checkpoint_path: str | Path,
    window_size: int,
    stride: int,
    batch_size: int,
    noise_draws: int,
    device: str,
    prior,
    converted_dir: str | Path | None,
) -> tuple[GroupScoreResult, dict[str, object]]:
    del checkpoint_path
    window_batches = []
    source_paths = []
    converted_paths = []

    for input_path in input_paths:
        windows, meta = load_motion_windows(
            input_path,
            dataset_label=label,
            window_size=window_size,
            stride=stride,
            converted_dir=converted_dir,
        )
        window_batches.append(windows)
        source_paths.append(str(meta["source_path"]))
        if meta["converted_path"] is not None:
            converted_paths.append(str(meta["converted_path"]))

    if not window_batches:
        raise ValueError(f"{label}_paths must be non-empty")

    group_result = score_motion_windows(
        np.concatenate(window_batches, axis=0),
        prior=prior,
        batch_size=batch_size,
        noise_draws=noise_draws,
        device=device,
        seed=0,
        label=label,
        source_paths=source_paths,
    )
    return group_result, {
        "source_paths": source_paths,
        "converted_paths": converted_paths,
    }


def _summary_to_dict(summary, *, extra: dict[str, object]) -> dict[str, object]:
    payload = asdict(summary)
    payload["per_timestep_raw_mse_mean"] = {
        str(timestep): float(value) for timestep, value in summary.per_timestep_raw_mse_mean.items()
    }
    payload.update(extra)
    return payload


def evaluate_smp_reward(
    *,
    checkpoint_path: str | Path,
    positive_paths: list[str | Path],
    negative_paths: list[str | Path],
    policy_paths: list[str | Path],
    window_size: int,
    stride: int = 1,
    batch_size: int = 128,
    noise_draws: int = 4,
    device: str = "cpu",
    converted_dir: str | Path | None = None,
    output_json: str | Path | None = None,
    style_name: str | None = None,
    style_id: int | None = None,
    timesteps_k: list[int] | None = None,
    reward_scale: float | None = None,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    resolved_converted_dir = None if converted_dir is None else Path(converted_dir)
    if resolved_converted_dir is None:
        resolved_converted_dir = _default_converted_dir(checkpoint_path, output_json)

    prior = load_prior_bundle(
        checkpoint_path,
        device=device,
        style_name=style_name,
        style_id=style_id,
        timesteps_k=timesteps_k,
        reward_scale=reward_scale,
    )

    scored_groups = []
    metadata_by_label = {}
    for label, paths, group_converted_dir in (
        ("positive", list(positive_paths), resolved_converted_dir / "positive"),
        ("negative", list(negative_paths), resolved_converted_dir / "negative"),
        ("policy", list(policy_paths), None),
    ):
        group_result, group_meta = _score_group(
            label=label,
            input_paths=paths,
            checkpoint_path=checkpoint_path,
            window_size=window_size,
            stride=stride,
            batch_size=batch_size,
            noise_draws=noise_draws,
            device=device,
            prior=prior,
            converted_dir=group_converted_dir,
        )
        scored_groups.append(group_result)
        metadata_by_label[label] = group_meta

    ordered_group_results = sorted(
        scored_groups,
        key=lambda group: _REPORT_GROUP_ORDER.index(group.label) if group.label in _REPORT_GROUP_ORDER else len(_REPORT_GROUP_ORDER),
    )
    summaries = normalize_and_summarize_group_scores(ordered_group_results, reward_scale=prior.reward_scale)

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "timesteps_k": [int(timestep) for timestep in prior.timesteps_k],
        "reward_scale": float(prior.reward_scale),
        "style_name": prior.style_name,
        "style_id": prior.style_id,
        "window_size": int(window_size),
        "stride": int(stride),
        "batch_size": int(batch_size),
        "noise_draws": int(noise_draws),
        "groups": {},
    }
    for label in _REPORT_GROUP_ORDER:
        if label not in summaries:
            continue
        payload["groups"][label] = _summary_to_dict(summaries[label], extra=metadata_by_label[label])

    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return payload


def format_evaluation_report(summary: dict[str, object]) -> str:
    group_payloads = summary["groups"]
    table_payload = {
        label: types.SimpleNamespace(**group_payloads[label]) for label in _REPORT_GROUP_ORDER if label in group_payloads
    }
    lines = [
        f"checkpoint_path: {summary['checkpoint_path']}",
        f"timesteps_k: {summary['timesteps_k']}",
        f"reward_scale: {summary['reward_scale']:.6f}",
        f"window_size: {summary['window_size']}",
        f"stride: {summary['stride']}",
        f"batch_size: {summary['batch_size']}",
        f"noise_draws: {summary['noise_draws']}",
        "",
        format_group_summary_table(table_payload),
        "",
        format_group_detail_lines(table_payload),
    ]
    for label in _REPORT_GROUP_ORDER:
        if label not in group_payloads:
            continue
        converted_paths = group_payloads[label].get("converted_paths", [])
        if converted_paths:
            lines.append(f"{label}_converted_paths: {converted_paths}")
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="比较 positive / policy / negative 三组动作窗口的 SMP reward。")
    parser.add_argument("--checkpoint", required=True, help="SMP prior checkpoint 路径。")
    parser.add_argument("--positive", nargs="+", required=True, help="原始 walk 等正确动作数据集 npz。")
    parser.add_argument("--negative", nargs="+", required=True, help="原始非目标动作数据集 npz。")
    parser.add_argument("--policy", nargs="+", required=True, help="通过 rollout 收集的 policy window 数据集 npz。")
    parser.add_argument("--window-size", required=True, type=int, help="SMP 观测窗口长度。")
    parser.add_argument("--stride", type=int, default=1, help="原始数据转窗口时的滑窗步长。")
    parser.add_argument("--batch-size", type=int, default=128, help="离线打分 batch size。")
    parser.add_argument("--noise-draws", type=int, default=4, help="每个 timestep 的噪声重采样次数。")
    parser.add_argument("--device", default="cpu", help="打分设备，例如 cpu 或 cuda:0。")
    parser.add_argument("--converted-dir", default=None, help="保存 positive / negative 转换后 frames 的目录。")
    parser.add_argument("--output-json", default=None, help="可选：把汇总结果写入 JSON。")
    parser.add_argument("--style-name", default=None, help="多风格 prior 的 style 名称。")
    parser.add_argument("--style-id", type=int, default=None, help="多风格 prior 的 style id。")
    parser.add_argument("--timesteps-k", type=int, nargs="+", default=None, help="覆盖 checkpoint 中的奖励时间步集合。")
    parser.add_argument("--reward-scale", type=float, default=None, help="覆盖 checkpoint 中的 reward scale。")
    return parser


def main():
    args = _build_argparser().parse_args()
    summary = evaluate_smp_reward(
        checkpoint_path=args.checkpoint,
        positive_paths=args.positive,
        negative_paths=args.negative,
        policy_paths=args.policy,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        noise_draws=args.noise_draws,
        device=args.device,
        converted_dir=args.converted_dir,
        output_json=args.output_json,
        style_name=args.style_name,
        style_id=args.style_id,
        timesteps_k=args.timesteps_k,
        reward_scale=args.reward_scale,
    )
    print(format_evaluation_report(summary))
    if args.output_json is not None:
        print(f"\nsummary_json: {Path(args.output_json)}")


if __name__ == "__main__":
    main()
