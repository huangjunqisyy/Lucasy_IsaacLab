#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PATH_BOOTSTRAP = _load_module(
    "isaaclab_smp_path_bootstrap_prior_audit",
    Path(__file__).resolve().with_name("path_bootstrap.py"),
)
_PATH_BOOTSTRAP.prepend_local_source_paths(__file__, package_names=("isaaclab", "isaaclab_rl", "isaaclab_tasks"))
_REPO_ROOT = _PATH_BOOTSTRAP.find_repo_root(__file__)
try:
    _NESTED_RSL_RL_ROOT = _PATH_BOOTSTRAP.find_nested_rsl_rl_repo_root(__file__)
except FileNotFoundError:
    _RSL_RL_CANDIDATES = [
        Path(os.environ.get("LUCAS_RSL_RL_ROOT", "")),
        Path("/home/hjqsyy/lucas_rsl_rl"),
    ]
    _NESTED_RSL_RL_ROOT = next(
        (candidate for candidate in _RSL_RL_CANDIDATES if (candidate / "rsl_rl" / "__init__.py").exists()),
        None,
    )
    if _NESTED_RSL_RL_ROOT is None:
        raise
if str(_NESTED_RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NESTED_RSL_RL_ROOT))

_SMP_DIAGNOSTICS = _load_module(
    "isaaclab_smp_prior_audit_diagnostics",
    _REPO_ROOT / "source" / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "smp_diagnostics.py",
)

from rsl_rl.diffusion import SMPDiffusionSampler  # noqa: E402

GroupScoreResult = _SMP_DIAGNOSTICS.GroupScoreResult
format_group_detail_lines = _SMP_DIAGNOSTICS.format_group_detail_lines
format_group_summary_table = _SMP_DIAGNOSTICS.format_group_summary_table
load_motion_windows = _SMP_DIAGNOSTICS.load_motion_windows
load_prior_bundle = _SMP_DIAGNOSTICS.load_prior_bundle
normalize_and_summarize_group_scores = _SMP_DIAGNOSTICS.normalize_and_summarize_group_scores
score_motion_windows = _SMP_DIAGNOSTICS.score_motion_windows


_G1_192D_BLOCKS = {
    "base_lin_vel_b": (0, 3),
    "base_ang_vel_b": (3, 6),
    "joint_rot6d_rel": (6, 180),
    "ee_pos_b": (180, 192),
}


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1.0e-8:
        return 1.0 if abs(float(numerator)) < 1.0e-8 else float("inf")
    return float(numerator) / float(denominator)


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
        resolved_paths.append(candidate.resolve())
    if not resolved_paths:
        raise ValueError(f"Manifest contains no dataset paths: {path}")
    return resolved_paths


def expand_dataset_paths(paths: list[str | Path]) -> list[Path]:
    expanded = []
    for path_like in paths:
        path = Path(path_like).expanduser().resolve()
        if path.suffix.lower() == ".json":
            expanded.extend(_resolve_dataset_manifest(path))
        else:
            expanded.append(path)
    return expanded


def _subsample_windows(windows: np.ndarray, max_windows: int | None, seed: int) -> np.ndarray:
    if max_windows is None or max_windows <= 0 or windows.shape[0] <= max_windows:
        return windows
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(windows.shape[0], size=int(max_windows), replace=False)
    return windows[np.sort(indices)]


def load_group_windows(
    paths: list[str | Path],
    *,
    label: str,
    window_size: int,
    stride: int,
    max_windows: int | None,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    window_batches = []
    source_paths = []
    for path in expand_dataset_paths(paths):
        windows, meta = load_motion_windows(path, dataset_label=label, window_size=window_size, stride=stride)
        window_batches.append(windows)
        source_paths.append(str(meta["source_path"]))
    if not window_batches:
        raise ValueError(f"No windows loaded for group '{label}'")
    merged = np.concatenate(window_batches, axis=0).astype(np.float32)
    return _subsample_windows(merged, max_windows=max_windows, seed=seed), source_paths


def compute_window_statistics(windows: np.ndarray, block_slices: dict[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, F), got {tuple(windows.shape)}")
    if windows.shape[0] == 0:
        raise ValueError("windows must be non-empty")

    flattened = windows.reshape(-1, windows.shape[-1])
    feature_mean = flattened.mean(axis=0)
    feature_std = flattened.std(axis=0)
    diff = np.diff(windows, axis=1)
    diff_norm = np.linalg.norm(diff, axis=-1) if diff.shape[1] > 0 else np.zeros((windows.shape[0], 0), dtype=np.float32)
    acc = np.diff(diff, axis=1)
    acc_norm = np.linalg.norm(acc, axis=-1) if acc.shape[1] > 0 else np.zeros((windows.shape[0], 0), dtype=np.float32)

    if block_slices is None:
        block_slices = _G1_192D_BLOCKS if windows.shape[-1] == 192 else {}

    block_stats = {}
    for block_name, (start, end) in block_slices.items():
        if start < 0 or end > windows.shape[-1] or start >= end:
            continue
        block = windows[..., start:end]
        block_diff = np.diff(block, axis=1)
        block_stats[block_name] = {
            "dim": int(end - start),
            "mean_abs_mean": float(np.abs(block.reshape(-1, end - start).mean(axis=0)).mean()),
            "std_mean": float(block.reshape(-1, end - start).std(axis=0).mean()),
            "temporal_diff_l2_mean": float(np.linalg.norm(block_diff, axis=-1).mean()) if block_diff.shape[1] > 0 else 0.0,
        }

    return {
        "num_windows": int(windows.shape[0]),
        "window_size": int(windows.shape[1]),
        "feature_dim": int(windows.shape[2]),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_mean_abs_mean": float(np.abs(feature_mean).mean()),
        "feature_std_mean": float(feature_std.mean()),
        "feature_std_max": float(feature_std.max()),
        "feature_abs_p99": float(np.percentile(np.abs(flattened), 99)),
        "temporal_diff_l2_mean": float(diff_norm.mean()) if diff_norm.size else 0.0,
        "temporal_diff_l2_std": float(diff_norm.std()) if diff_norm.size else 0.0,
        "temporal_acc_l2_mean": float(acc_norm.mean()) if acc_norm.size else 0.0,
        "temporal_acc_l2_std": float(acc_norm.std()) if acc_norm.size else 0.0,
        "blocks": block_stats,
    }


def compare_window_statistics(reference_stats: dict[str, Any], generated_stats: dict[str, Any]) -> dict[str, Any]:
    ref_mean = np.asarray(reference_stats["feature_mean"], dtype=np.float32)
    gen_mean = np.asarray(generated_stats["feature_mean"], dtype=np.float32)
    ref_std = np.asarray(reference_stats["feature_std"], dtype=np.float32)
    gen_std = np.asarray(generated_stats["feature_std"], dtype=np.float32)
    if ref_mean.shape != gen_mean.shape:
        raise ValueError(f"Feature mean shapes differ: {ref_mean.shape} vs {gen_mean.shape}")
    if ref_std.shape != gen_std.shape:
        raise ValueError(f"Feature std shapes differ: {ref_std.shape} vs {gen_std.shape}")

    block_comparison = {}
    for block_name, ref_block in dict(reference_stats.get("blocks", {})).items():
        gen_block = dict(generated_stats.get("blocks", {})).get(block_name)
        if gen_block is None:
            continue
        block_comparison[block_name] = {
            "std_mean_ratio": _safe_ratio(gen_block["std_mean"], ref_block["std_mean"]),
            "temporal_diff_l2_ratio": _safe_ratio(
                gen_block["temporal_diff_l2_mean"],
                ref_block["temporal_diff_l2_mean"],
            ),
        }

    return {
        "feature_mean_l2": float(np.sqrt(np.mean((gen_mean - ref_mean) ** 2))),
        "feature_std_l2": float(np.sqrt(np.mean((gen_std - ref_std) ** 2))),
        "feature_std_ratio": _safe_ratio(generated_stats["feature_std_mean"], reference_stats["feature_std_mean"]),
        "feature_abs_p99_ratio": _safe_ratio(generated_stats["feature_abs_p99"], reference_stats["feature_abs_p99"]),
        "temporal_diff_l2_ratio": _safe_ratio(
            generated_stats["temporal_diff_l2_mean"],
            reference_stats["temporal_diff_l2_mean"],
        ),
        "temporal_acc_l2_ratio": _safe_ratio(
            generated_stats["temporal_acc_l2_mean"],
            reference_stats["temporal_acc_l2_mean"],
        ),
        "blocks": block_comparison,
    }


def _group_result_to_dict(summary) -> dict[str, Any]:
    payload = asdict(summary)
    payload["per_timestep_raw_mse_mean"] = {
        str(timestep): float(value) for timestep, value in summary.per_timestep_raw_mse_mean.items()
    }
    return payload


def _score_windows(label: str, windows: np.ndarray, prior, *, batch_size: int, noise_draws: int, device: str, source_paths: list[str]):
    return score_motion_windows(
        windows,
        prior=prior,
        batch_size=batch_size,
        noise_draws=noise_draws,
        device=device,
        seed=0,
        label=label,
        source_paths=source_paths,
    )


def generate_prior_windows(
    prior,
    *,
    num_generated: int,
    batch_size: int,
    device: str,
    seed: int,
    guidance_scale: float,
) -> np.ndarray:
    if num_generated <= 0:
        raise ValueError(f"num_generated must be positive, got {num_generated}")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model_cfg = prior.checkpoint["model_cfg"]
    sampler = SMPDiffusionSampler(
        model=prior.model,
        num_diffusion_steps=int(model_cfg["num_diffusion_steps"]),
        feature_dim=int(model_cfg["feature_dim"]),
        window_size=int(model_cfg["window_size"]),
        device=device,
    )
    batches = []
    remaining = int(num_generated)
    while remaining > 0:
        current_batch = min(int(batch_size), remaining)
        sample = sampler.sample(
            batch_size=current_batch,
            style_id=prior.style_id,
            guidance_scale=float(guidance_scale),
        )
        batches.append(sample.detach().float().cpu().numpy())
        remaining -= current_batch
    return np.concatenate(batches, axis=0).astype(np.float32)


def build_pass_fail_flags(
    *,
    group_summaries: dict[str, dict[str, Any]],
    generated_comparison: dict[str, Any] | None,
    max_val_train_mse_ratio: float,
    max_generated_mean_l2: float,
    max_generated_std_l2: float,
    min_temporal_ratio: float,
    max_temporal_ratio: float,
) -> dict[str, bool | None]:
    train = group_summaries.get("train")
    val = group_summaries.get("val")
    negative = group_summaries.get("negative")
    flags: dict[str, bool | None] = {
        "val_close_to_train": None,
        "negative_worse_than_val": None,
        "generated_mean_close": None,
        "generated_std_close": None,
        "generated_temporal_diff_reasonable": None,
        "generated_temporal_acc_reasonable": None,
    }

    if train is not None and val is not None:
        flags["val_close_to_train"] = (
            float(val["raw_noise_mse_mean"]) <= float(train["raw_noise_mse_mean"]) * float(max_val_train_mse_ratio)
        )
    if val is not None and negative is not None:
        flags["negative_worse_than_val"] = float(negative["raw_noise_mse_mean"]) > float(val["raw_noise_mse_mean"])

    if generated_comparison is not None:
        flags["generated_mean_close"] = float(generated_comparison["feature_mean_l2"]) <= float(max_generated_mean_l2)
        flags["generated_std_close"] = float(generated_comparison["feature_std_l2"]) <= float(max_generated_std_l2)
        flags["generated_temporal_diff_reasonable"] = (
            float(min_temporal_ratio)
            <= float(generated_comparison["temporal_diff_l2_ratio"])
            <= float(max_temporal_ratio)
        )
        flags["generated_temporal_acc_reasonable"] = (
            float(min_temporal_ratio)
            <= float(generated_comparison["temporal_acc_l2_ratio"])
            <= float(max_temporal_ratio)
        )

    active_flags = [value for value in flags.values() if value is not None]
    flags["overall"] = all(active_flags) if active_flags else None
    return flags


def audit_smp_prior_quality(
    *,
    checkpoint_path: str | Path,
    train_paths: list[str | Path],
    val_paths: list[str | Path] | None,
    negative_paths: list[str | Path] | None,
    policy_paths: list[str | Path] | None,
    window_size: int,
    stride: int,
    batch_size: int,
    noise_draws: int,
    max_windows_per_group: int | None,
    num_generated: int,
    generation_batch_size: int,
    device: str,
    style_name: str | None,
    style_id: int | None,
    reward_scale: float | None,
    guidance_scale: float,
    output_json: str | Path | None,
    generated_output: str | Path | None,
    seed: int,
    max_val_train_mse_ratio: float,
    max_generated_mean_l2: float,
    max_generated_std_l2: float,
    min_temporal_ratio: float,
    max_temporal_ratio: float,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    prior = load_prior_bundle(
        checkpoint_path,
        device=device,
        style_name=style_name,
        style_id=style_id,
        reward_scale=reward_scale,
    )

    group_windows: dict[str, np.ndarray] = {}
    group_sources: dict[str, list[str]] = {}
    for label, paths in (
        ("train", train_paths),
        ("val", val_paths),
        ("negative", negative_paths),
        ("policy", policy_paths),
    ):
        if not paths:
            continue
        windows, sources = load_group_windows(
            list(paths),
            label=label,
            window_size=window_size,
            stride=stride,
            max_windows=max_windows_per_group,
            seed=seed + len(group_windows),
        )
        group_windows[label] = windows
        group_sources[label] = sources

    if "train" not in group_windows:
        raise ValueError("train_paths must load at least one group of windows")

    generated_windows = generate_prior_windows(
        prior,
        num_generated=num_generated,
        batch_size=generation_batch_size,
        device=device,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    if generated_output is not None:
        generated_path = Path(generated_output)
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            generated_path,
            windows=generated_windows,
            window_size=np.array([generated_windows.shape[1]], dtype=np.int64),
            feature_dim=np.array([generated_windows.shape[2]], dtype=np.int64),
            source_name=np.asarray(["generated_prior"], dtype=np.str_),
        )

    group_windows["generated"] = generated_windows
    group_sources["generated"] = ["generated_prior"]

    score_results = [
        _score_windows(
            label,
            windows,
            prior,
            batch_size=batch_size,
            noise_draws=noise_draws,
            device=device,
            source_paths=group_sources[label],
        )
        for label, windows in group_windows.items()
    ]
    summaries = normalize_and_summarize_group_scores(score_results, reward_scale=prior.reward_scale, reference_label="train")
    group_summaries = {label: _group_result_to_dict(summary) for label, summary in summaries.items()}

    train_stats = compute_window_statistics(group_windows["train"])
    generated_stats = compute_window_statistics(generated_windows)
    generated_comparison = compare_window_statistics(train_stats, generated_stats)
    distribution_stats = {
        "train": train_stats,
        "generated": generated_stats,
        "generated_vs_train": generated_comparison,
    }
    for label in ("val", "negative", "policy"):
        if label in group_windows:
            distribution_stats[label] = compute_window_statistics(group_windows[label])

    checkpoint = prior.checkpoint
    model_cfg = dict(checkpoint.get("model_cfg", {}))
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "model_cfg": model_cfg,
        "timesteps_k": [int(timestep) for timestep in prior.timesteps_k],
        "reward_scale": float(prior.reward_scale),
        "style_name": prior.style_name,
        "style_id": prior.style_id,
        "window_size_arg": int(window_size),
        "stride": int(stride),
        "max_windows_per_group": None if max_windows_per_group is None else int(max_windows_per_group),
        "num_generated": int(num_generated),
        "device": str(device),
    }
    flags = build_pass_fail_flags(
        group_summaries=group_summaries,
        generated_comparison=generated_comparison,
        max_val_train_mse_ratio=max_val_train_mse_ratio,
        max_generated_mean_l2=max_generated_mean_l2,
        max_generated_std_l2=max_generated_std_l2,
        min_temporal_ratio=min_temporal_ratio,
        max_temporal_ratio=max_temporal_ratio,
    )

    payload = {
        "metadata": metadata,
        "groups": group_summaries,
        "distribution": distribution_stats,
        "pass_fail": flags,
    }
    if output_json is not None:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(_as_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return _as_jsonable(payload)


def format_audit_report(summary: dict[str, Any]) -> str:
    groups = {label: type("Summary", (), payload) for label, payload in summary["groups"].items()}
    ordered_labels = [label for label in ("train", "val", "policy", "negative", "generated") if label in groups]
    ordered_groups = {label: groups[label] for label in ordered_labels}
    lines = [
        f"checkpoint_path: {summary['metadata']['checkpoint_path']}",
        f"model_cfg: {summary['metadata']['model_cfg']}",
        f"timesteps_k: {summary['metadata']['timesteps_k']}",
        f"style: name={summary['metadata']['style_name']} id={summary['metadata']['style_id']}",
        "",
        format_group_summary_table(ordered_groups),
        "",
        format_group_detail_lines(ordered_groups),
        "",
        "[generated_vs_train_distribution]",
    ]
    comparison = summary["distribution"]["generated_vs_train"]
    for key in (
        "feature_mean_l2",
        "feature_std_l2",
        "feature_std_ratio",
        "feature_abs_p99_ratio",
        "temporal_diff_l2_ratio",
        "temporal_acc_l2_ratio",
    ):
        lines.append(f"  {key}={float(comparison[key]):.6f}")
    lines.append("")
    lines.append("[pass_fail]")
    for key, value in summary["pass_fail"].items():
        lines.append(f"  {key}={value}")
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="审查 SMP diffusion prior 是否学到输入运动数据分布。")
    parser.add_argument("--checkpoint", required=True, help="SMP prior checkpoint 路径。")
    parser.add_argument("--train", nargs="+", required=True, help="训练/参考 walk 数据 npz 或 manifest json。")
    parser.add_argument("--val", nargs="+", default=None, help="留出的验证 walk 数据 npz 或 manifest json。")
    parser.add_argument("--negative", nargs="+", default=None, help="非目标动作 npz 或 manifest json，用于判别性检查。")
    parser.add_argument("--policy", nargs="+", default=None, help="policy rollout windows npz，可选。")
    parser.add_argument("--window-size", type=int, default=10, help="SMP 窗口长度。")
    parser.add_argument("--stride", type=int, default=1, help="frames 转 windows 的滑窗步长。")
    parser.add_argument("--batch-size", type=int, default=128, help="prior 打分 batch size。")
    parser.add_argument("--noise-draws", type=int, default=4, help="每个 timestep 重采样噪声次数。")
    parser.add_argument("--max-windows-per-group", type=int, default=4096, help="每组最多抽样多少窗口；<=0 表示全量。")
    parser.add_argument("--num-generated", type=int, default=512, help="从 prior 生成多少个窗口做分布审查。")
    parser.add_argument("--generation-batch-size", type=int, default=64, help="生成窗口时的 batch size。")
    parser.add_argument("--device", default="cpu", help="运行设备，例如 cpu 或 cuda:0。")
    parser.add_argument("--style-name", default=None, help="条件 prior 的目标 style 名。")
    parser.add_argument("--style-id", type=int, default=None, help="条件 prior 的目标 style id。")
    parser.add_argument("--reward-scale", type=float, default=None, help="覆盖 reward scale。")
    parser.add_argument("--guidance-scale", type=float, default=1.0, help="生成样本时的 CFG guidance scale。")
    parser.add_argument("--seed", type=int, default=0, help="抽样和生成随机种子。")
    parser.add_argument("--output-json", default=None, help="保存完整审查结果 JSON。")
    parser.add_argument("--generated-output", default=None, help="可选：保存 prior 生成的 windows npz。")
    parser.add_argument("--max-val-train-mse-ratio", type=float, default=1.5, help="val/train MSE 最大容忍比例。")
    parser.add_argument("--max-generated-mean-l2", type=float, default=1.0, help="生成/真实 feature 均值 RMS 差阈值。")
    parser.add_argument("--max-generated-std-l2", type=float, default=1.0, help="生成/真实 feature 标准差 RMS 差阈值。")
    parser.add_argument("--min-temporal-ratio", type=float, default=0.5, help="生成 temporal 指标相对真实的最小比例。")
    parser.add_argument("--max-temporal-ratio", type=float, default=2.0, help="生成 temporal 指标相对真实的最大比例。")
    return parser


def main():
    args = _build_argparser().parse_args()
    max_windows = None if args.max_windows_per_group <= 0 else int(args.max_windows_per_group)
    summary = audit_smp_prior_quality(
        checkpoint_path=args.checkpoint,
        train_paths=args.train,
        val_paths=args.val,
        negative_paths=args.negative,
        policy_paths=args.policy,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        noise_draws=args.noise_draws,
        max_windows_per_group=max_windows,
        num_generated=args.num_generated,
        generation_batch_size=args.generation_batch_size,
        device=args.device,
        style_name=args.style_name,
        style_id=args.style_id,
        reward_scale=args.reward_scale,
        guidance_scale=args.guidance_scale,
        output_json=args.output_json,
        generated_output=args.generated_output,
        seed=args.seed,
        max_val_train_mse_ratio=args.max_val_train_mse_ratio,
        max_generated_mean_l2=args.max_generated_mean_l2,
        max_generated_std_l2=args.max_generated_std_l2,
        min_temporal_ratio=args.min_temporal_ratio,
        max_temporal_ratio=args.max_temporal_ratio,
    )
    print(format_audit_report(summary))
    if args.output_json is not None:
        print(f"\nsummary_json: {Path(args.output_json)}")
    if args.generated_output is not None:
        print(f"generated_windows: {Path(args.generated_output)}")


if __name__ == "__main__":
    main()
