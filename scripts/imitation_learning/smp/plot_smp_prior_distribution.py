#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_PATH_BOOTSTRAP = _load_module(
    "isaaclab_smp_path_bootstrap_distribution_plots",
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

_AUDIT = _load_module(
    "isaaclab_smp_prior_distribution_audit",
    Path(__file__).resolve().with_name("audit_smp_prior_quality.py"),
)


G1_EE_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]
G1_192D_BLOCKS = {
    "base_lin_vel_b": (0, 3),
    "base_ang_vel_b": (3, 6),
    "joint_rot6d_rel": (6, 180),
    "ee_pos_b": (180, 192),
}
AXES = ("x", "y", "z")


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1.0e-8:
        return 1.0 if abs(float(numerator)) < 1.0e-8 else float("inf")
    return float(numerator) / float(denominator)


def _flatten(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(-1)


def _temporal_diff_l2(windows: np.ndarray) -> np.ndarray:
    diff = np.diff(windows, axis=1)
    if diff.shape[1] == 0:
        return np.zeros((windows.shape[0],), dtype=np.float32)
    return np.linalg.norm(diff, axis=-1).reshape(-1).astype(np.float32)


def _temporal_acc_l2(windows: np.ndarray) -> np.ndarray:
    diff = np.diff(windows, axis=1)
    acc = np.diff(diff, axis=1)
    if acc.shape[1] == 0:
        return np.zeros((windows.shape[0],), dtype=np.float32)
    return np.linalg.norm(acc, axis=-1).reshape(-1).astype(np.float32)


def extract_named_series(windows: np.ndarray) -> dict[str, np.ndarray]:
    """把 G1 192D windows 拆成可解释的一维分布序列。"""
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, F), got {tuple(windows.shape)}")
    if windows.shape[-1] != 192:
        raise ValueError(f"This script expects G1 192D SMP features, got feature_dim={windows.shape[-1]}")

    series: dict[str, np.ndarray] = {}
    base_lin = windows[..., 0:3]
    base_ang = windows[..., 3:6]
    joint = windows[..., 6:180]
    ee = windows[..., 180:192].reshape(*windows.shape[:2], len(G1_EE_NAMES), 3)

    for axis_index, axis_name in enumerate(AXES):
        series[f"base_lin_vel_b/{axis_name}"] = _flatten(base_lin[..., axis_index])
        series[f"base_ang_vel_b/{axis_name}"] = _flatten(base_ang[..., axis_index])

    for ee_index, ee_name in enumerate(G1_EE_NAMES):
        for axis_index, axis_name in enumerate(AXES):
            series[f"ee_pos_b/{ee_name}/{axis_name}"] = _flatten(ee[..., ee_index, axis_index])

    for block_name, (start, end) in G1_192D_BLOCKS.items():
        block = windows[..., start:end]
        series[f"norm/{block_name}"] = np.linalg.norm(block, axis=-1).reshape(-1).astype(np.float32)
        series[f"temporal_diff/{block_name}"] = _temporal_diff_l2(block)
        series[f"temporal_acc/{block_name}"] = _temporal_acc_l2(block)

    series["temporal/diff_l2"] = _temporal_diff_l2(windows)
    series["temporal/acc_l2"] = _temporal_acc_l2(windows)

    for dim in range(joint.shape[-1]):
        series[f"joint_rot6d_rel/dim_{dim:03d}"] = _flatten(joint[..., dim])
    return series


def _describe(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {key: 0.0 for key in ("mean", "std", "p01", "p10", "p50", "p90", "p99")}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
    }


def build_series_report(reference_series: dict[str, np.ndarray], generated_series: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(set(reference_series) & set(generated_series)):
        ref_stats = _describe(reference_series[name])
        gen_stats = _describe(generated_series[name])
        rows.append(
            {
                "name": name,
                "reference_mean": ref_stats["mean"],
                "generated_mean": gen_stats["mean"],
                "generated_minus_reference_mean": gen_stats["mean"] - ref_stats["mean"],
                "generated_mean_ratio": _safe_ratio(gen_stats["mean"], ref_stats["mean"]),
                "reference_std": ref_stats["std"],
                "generated_std": gen_stats["std"],
                "generated_minus_reference_std": gen_stats["std"] - ref_stats["std"],
                "generated_std_ratio": _safe_ratio(gen_stats["std"], ref_stats["std"]),
                "reference_p10": ref_stats["p10"],
                "generated_p10": gen_stats["p10"],
                "reference_p50": ref_stats["p50"],
                "generated_p50": gen_stats["p50"],
                "reference_p90": ref_stats["p90"],
                "generated_p90": gen_stats["p90"],
                "reference_p99": ref_stats["p99"],
                "generated_p99": gen_stats["p99"],
            }
        )
    return rows


def select_plot_series(keys: list[str] | tuple[str, ...], limit: int = 20) -> list[str]:
    priority = [
        "base_lin_vel_b/x",
        "base_lin_vel_b/y",
        "base_lin_vel_b/z",
        "base_ang_vel_b/x",
        "base_ang_vel_b/y",
        "base_ang_vel_b/z",
        "ee_pos_b/left_ankle_roll_link/z",
        "ee_pos_b/right_ankle_roll_link/z",
        "ee_pos_b/left_wrist_roll_link/z",
        "ee_pos_b/right_wrist_roll_link/z",
        "norm/base_lin_vel_b",
        "norm/base_ang_vel_b",
        "norm/joint_rot6d_rel",
        "norm/ee_pos_b",
        "temporal_diff/base_lin_vel_b",
        "temporal_diff/base_ang_vel_b",
        "temporal_diff/joint_rot6d_rel",
        "temporal_diff/ee_pos_b",
        "temporal/diff_l2",
        "temporal/acc_l2",
    ]
    key_set = set(keys)
    selected = [name for name in priority if name in key_set]
    if len(selected) < limit:
        selected.extend([name for name in sorted(key_set) if name not in selected and not name.startswith("joint_rot6d_rel/")])
    return selected[:limit]


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_generated_windows(path: str | Path) -> np.ndarray:
    with np.load(Path(path)) as data:
        if "windows" not in data.files:
            raise ValueError(f"Generated npz must contain a 'windows' array: {path}")
        windows = np.asarray(data["windows"], dtype=np.float32)
    if windows.ndim == 4:
        windows = windows.reshape(-1, windows.shape[-2], windows.shape[-1])
    if windows.ndim != 3:
        raise ValueError(f"Expected generated windows shape (N, W, F), got {tuple(windows.shape)}")
    return windows


def _subsample_windows(windows: np.ndarray, max_windows: int | None, seed: int) -> np.ndarray:
    if max_windows is None or max_windows <= 0 or windows.shape[0] <= max_windows:
        return windows
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(windows.shape[0], size=int(max_windows), replace=False)
    return windows[np.sort(indices)]


def _save_histograms(reference_series: dict[str, np.ndarray], generated_series: dict[str, np.ndarray], output_path: Path, max_plots: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plot_names = select_plot_series(list(reference_series.keys() & generated_series.keys()), limit=max_plots)
    cols = 2
    rows = int(np.ceil(len(plot_names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, max(3, rows * 3)))
    axes = np.asarray(axes).reshape(-1)
    for axis, name in zip(axes, plot_names):
        ref_values = reference_series[name]
        gen_values = generated_series[name]
        axis.hist(ref_values, bins=60, density=False, alpha=0.55, label="reference")
        axis.hist(gen_values, bins=60, density=False, alpha=0.55, label="generated")
        axis.set_title(name)
        axis.grid(True, alpha=0.2)
    for axis in axes[len(plot_names) :]:
        axis.axis("off")
    axes[0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _save_sequence_examples(reference_windows: np.ndarray, generated_windows: np.ndarray, output_path: Path, num_examples: int) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    num_examples = min(int(num_examples), reference_windows.shape[0], generated_windows.shape[0])
    if num_examples <= 0:
        return False
    fig, axes = plt.subplots(num_examples, 4, figsize=(16, max(3, 3 * num_examples)), squeeze=False)
    for row in range(num_examples):
        ref = reference_windows[row]
        gen = generated_windows[row]
        panels = [
            ("base_lin_vel_b/x", ref[:, 0], gen[:, 0]),
            ("base_lin_vel_b/y", ref[:, 1], gen[:, 1]),
            ("base_ang_vel_b/z", ref[:, 5], gen[:, 5]),
            ("left/right foot z", ref[:, [182, 185]], gen[:, [182, 185]]),
        ]
        for col, (title, ref_values, gen_values) in enumerate(panels):
            axis = axes[row][col]
            if ref_values.ndim == 1:
                axis.plot(ref_values, label="ref", color="tab:blue")
                axis.plot(gen_values, label="gen", color="tab:orange")
            else:
                axis.plot(ref_values[:, 0], label="ref left", color="tab:blue")
                axis.plot(ref_values[:, 1], label="ref right", color="tab:cyan")
                axis.plot(gen_values[:, 0], label="gen left", color="tab:orange")
                axis.plot(gen_values[:, 1], label="gen right", color="tab:red")
            axis.set_title(f"example {row}: {title}")
            axis.grid(True, alpha=0.2)
    axes[0][0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def compare_prior_distribution(
    *,
    reference_paths: list[str | Path],
    generated_path: str | Path,
    output_dir: str | Path,
    window_size: int,
    stride: int,
    max_windows: int | None,
    seed: int,
    max_plots: int,
    num_sequence_examples: int,
) -> dict[str, Any]:
    reference_windows, reference_sources = _AUDIT.load_group_windows(
        list(reference_paths),
        label="reference",
        window_size=window_size,
        stride=stride,
        max_windows=max_windows,
        seed=seed,
    )
    generated_windows = _subsample_windows(_load_generated_windows(generated_path), max_windows=max_windows, seed=seed + 1)
    if reference_windows.shape[-1] != generated_windows.shape[-1]:
        raise ValueError(
            f"Feature dim mismatch: reference={reference_windows.shape[-1]}, generated={generated_windows.shape[-1]}"
        )
    if reference_windows.shape[1] != generated_windows.shape[1]:
        raise ValueError(
            f"Window size mismatch: reference={reference_windows.shape[1]}, generated={generated_windows.shape[1]}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_series = extract_named_series(reference_windows)
    generated_series = extract_named_series(generated_windows)
    rows = build_series_report(reference_series, generated_series)
    report = {
        "metadata": {
            "reference_sources": reference_sources,
            "generated_path": str(generated_path),
            "reference_shape": list(reference_windows.shape),
            "generated_shape": list(generated_windows.shape),
            "window_size": int(window_size),
            "stride": int(stride),
            "max_windows": None if max_windows is None else int(max_windows),
        },
        "block_stats": {
            "reference": _AUDIT.compute_window_statistics(reference_windows)["blocks"],
            "generated": _AUDIT.compute_window_statistics(generated_windows)["blocks"],
            "generated_vs_reference": _AUDIT.compare_window_statistics(
                _AUDIT.compute_window_statistics(reference_windows),
                _AUDIT.compute_window_statistics(generated_windows),
            )["blocks"],
        },
        "series": rows,
    }

    json_path = output_dir / "distribution_report.json"
    csv_path = output_dir / "series_report.csv"
    hist_path = output_dir / "series_histograms.png"
    sequence_path = output_dir / "sequence_examples.png"
    json_path.write_text(json.dumps(_as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, csv_path)
    histogram_saved = _save_histograms(reference_series, generated_series, hist_path, max_plots=max_plots)
    sequence_saved = _save_sequence_examples(reference_windows, generated_windows, sequence_path, num_sequence_examples)
    report["outputs"] = {
        "json": str(json_path),
        "csv": str(csv_path),
        "histograms": str(hist_path) if histogram_saved else None,
        "sequence_examples": str(sequence_path) if sequence_saved else None,
    }
    json_path.write_text(json.dumps(_as_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return _as_jsonable(report)


def format_console_report(report: dict[str, Any], top_k: int = 20) -> str:
    lines = [
        f"reference_shape: {report['metadata']['reference_shape']}",
        f"generated_shape: {report['metadata']['generated_shape']}",
        "",
        "[block generated/reference ratios]",
    ]
    block_ratios = report["block_stats"]["generated_vs_reference"]
    for name, values in block_ratios.items():
        lines.append(
            f"  {name}: std_mean_ratio={values['std_mean_ratio']:.4f}, "
            f"temporal_diff_l2_ratio={values['temporal_diff_l2_ratio']:.4f}"
        )

    lines.append("")
    lines.append("[selected series ratios]")
    selected = select_plot_series([row["name"] for row in report["series"]], limit=top_k)
    by_name = {row["name"]: row for row in report["series"]}
    for name in selected:
        row = by_name[name]
        lines.append(
            f"  {name}: mean_delta={row['generated_minus_reference_mean']:.4f}, "
            f"std_ratio={row['generated_std_ratio']:.4f}, "
            f"ref_p50={row['reference_p50']:.4f}, gen_p50={row['generated_p50']:.4f}"
        )

    lines.append("")
    lines.append("[outputs]")
    for key, value in report["outputs"].items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对比真实 G1 192D SMP windows 与 prior 生成 windows 的分布并画图。")
    parser.add_argument("--reference", nargs="+", required=True, help="真实数据 npz 或 manifest json。")
    parser.add_argument("--generated", required=True, help="audit 脚本保存的 generated windows npz。")
    parser.add_argument("--output-dir", required=True, help="输出 JSON/CSV/PNG 的目录。")
    parser.add_argument("--window-size", type=int, default=10, help="真实 frames 转 windows 的窗口长度。")
    parser.add_argument("--stride", type=int, default=1, help="真实 frames 转 windows 的滑窗步长。")
    parser.add_argument("--max-windows", type=int, default=4096, help="每组最多抽样窗口数；<=0 表示全量。")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子。")
    parser.add_argument("--max-plots", type=int, default=20, help="直方图最多画多少个一维序列。")
    parser.add_argument("--num-sequence-examples", type=int, default=6, help="画多少组窗口时序例子。")
    return parser


def main():
    args = _build_argparser().parse_args()
    max_windows = None if args.max_windows <= 0 else int(args.max_windows)
    report = compare_prior_distribution(
        reference_paths=args.reference,
        generated_path=args.generated,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        max_windows=max_windows,
        seed=args.seed,
        max_plots=args.max_plots,
        num_sequence_examples=args.num_sequence_examples,
    )
    print(format_console_report(report))


if __name__ == "__main__":
    main()
