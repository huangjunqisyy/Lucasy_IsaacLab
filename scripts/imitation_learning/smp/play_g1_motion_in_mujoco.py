#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_UTILS = _load_module(
    "isaaclab_smp_playback_utils_entry",
    Path(__file__).resolve().with_name("smp_mujoco_playback_utils.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MuJoCo viewer 中回放 G1 的 csv 或 npz 运动数据。")
    parser.add_argument("--input", required=True, help="单个 csv/npz 文件，或包含它们的目录。")
    parser.add_argument("--urdf", default=None, help="可选：显式指定 G1 URDF。")
    parser.add_argument("--mode", default="auto", choices=["auto", "raw", "frames", "features", "continuous", "sample", "window"], help="npz 优先读取模式。")
    parser.add_argument("--csv-fps", type=float, default=30.0, help="输入为 csv 时使用的 fps。")
    parser.add_argument("--sample-index", type=int, default=0, help="播放 samples_denormalized 的第几个样本。")
    parser.add_argument("--window-index", type=int, default=0, help="播放 windows 的第几个窗口。")
    parser.add_argument("--fps", type=float, default=None, help="强制覆盖回放 fps。")
    parser.add_argument("--default-height", type=float, default=0.74, help="192D 解码回放时使用的默认 pelvis 高度。")
    parser.add_argument("--loop", action="store_true", help="循环播放。")
    parser.add_argument("--hold", type=float, default=0.5, help="每段轨迹播完后停留秒数。")
    parser.add_argument("--max-files", type=int, default=0, help="目录模式下最多播放多少个文件。0 表示全部。")
    parser.add_argument("--dry-run", action="store_true", help="只打印摘要，不打开 viewer。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_files = _UTILS.collect_input_files(args.input)
    if args.max_files > 0:
        input_files = input_files[: args.max_files]

    trajectories = []
    for input_path in input_files:
        if input_path.suffix.lower() == ".csv":
            trajectory = _UTILS.load_csv_trajectory(input_path, fps=args.csv_fps)
        else:
            trajectory = _UTILS.load_npz_trajectory(
                npz_path=input_path,
                mode=args.mode,
                sample_index=args.sample_index,
                window_index=args.window_index,
                extractor=None,
                default_height=args.default_height,
                fps_override=args.fps,
            )
        trajectories.append(trajectory)

    for trajectory in trajectories:
        _UTILS.print_trajectory_summary(trajectory)

    if args.dry_run:
        return

    model = _UTILS.build_floating_model_from_urdf(_UTILS.resolve_g1_urdf_path(args.urdf))
    _UTILS.play_trajectories(model=model, trajectories=trajectories, loop=args.loop, hold=args.hold)


if __name__ == "__main__":
    main()
