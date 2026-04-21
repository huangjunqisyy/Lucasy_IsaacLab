#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

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


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLAYBACK_UTILS = _load_module(
    "isaaclab_smp_denoising_playback_utils",
    _THIS_DIR / "smp_mujoco_playback_utils.py",
)


def reconstruct_x0_from_eps(
    *,
    xt: torch.Tensor,
    t: torch.Tensor,
    eps_hat: torch.Tensor,
    scheduler,
) -> torch.Tensor:
    if xt.shape != eps_hat.shape:
        raise ValueError(f"xt and eps_hat must share the same shape, got {tuple(xt.shape)} and {tuple(eps_hat.shape)}")
    if t.ndim != 1 or t.shape[0] != xt.shape[0]:
        raise ValueError(f"t must have shape ({xt.shape[0]},), got {tuple(t.shape)}")

    alpha_bar = scheduler.alpha_bar.to(device=xt.device, dtype=xt.dtype)[t]
    alpha_bar = alpha_bar.view(-1, *([1] * (xt.ndim - 1)))
    return (xt - (1.0 - alpha_bar).sqrt() * eps_hat) / alpha_bar.sqrt().clamp_min(1.0e-8)


def build_noisy_window_bundle(
    *,
    x0: torch.Tensor,
    scheduler,
    timestep: int,
    seed: int,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    device = torch.device(device)
    x0 = x0.to(device=device, dtype=torch.float32)
    if x0.ndim != 3:
        raise ValueError(f"Expected x0 shape (B, W, F), got {tuple(x0.shape)}")
    if timestep < 0 or timestep >= scheduler.num_steps:
        raise ValueError(f"timestep must stay within [0, {scheduler.num_steps - 1}]")

    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(seed))
    eps = torch.randn(x0.shape, generator=generator, device=device, dtype=x0.dtype)
    t = torch.full((x0.shape[0],), int(timestep), device=device, dtype=torch.long)
    xt = scheduler.q_sample(x0, t, eps)
    return {"x0": x0, "eps": eps, "t": t, "xt": xt}


def stitch_overlapping_windows(windows: torch.Tensor, stride: int) -> torch.Tensor:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, F), got {tuple(windows.shape)}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    num_windows, window_size, feature_dim = windows.shape
    if num_windows <= 0:
        raise ValueError("windows must be non-empty")
    sequence_length = window_size + (num_windows - 1) * stride
    stitched = torch.zeros((sequence_length, feature_dim), dtype=windows.dtype, device=windows.device)
    counts = torch.zeros((sequence_length, 1), dtype=windows.dtype, device=windows.device)
    for window_idx in range(num_windows):
        start = window_idx * stride
        stop = start + window_size
        stitched[start:stop] += windows[window_idx]
        counts[start:stop] += 1.0
    return stitched / counts.clamp_min(1.0)


def _load_prior_bundle(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    style_name: str | None,
    style_id: int | None,
):
    diagnostics = _load_module(
        "isaaclab_smp_denoising_diagnostics",
        _REPO_ROOT / "source" / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "smp_diagnostics.py",
    )
    return diagnostics.load_prior_bundle(
        checkpoint_path=checkpoint_path,
        device=device,
        style_name=style_name,
        style_id=style_id,
    )


def _select_windows(sequence: np.ndarray, *, window_size: int, stride: int, window_index: int, num_windows: int) -> np.ndarray:
    return _PLAYBACK_UTILS.select_window_batch(
        sequence,
        window_size=window_size,
        stride=stride,
        window_index=window_index,
        num_windows=num_windows,
    )


def _predict_eps(prior, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        if prior.style_id is None:
            return prior.model(xt, t)
        style_id_tensor = torch.full((xt.shape[0],), int(prior.style_id), device=xt.device, dtype=torch.long)
        return prior.model(xt, t, style_id=style_id_tensor)


def _compute_summary(original: torch.Tensor, xt: torch.Tensor, reconstructed: torch.Tensor, eps: torch.Tensor, eps_hat: torch.Tensor) -> dict[str, float]:
    return {
        "feature_noisy_mse": float((xt - original).pow(2).mean().item()),
        "feature_reconstruction_mse": float((reconstructed - original).pow(2).mean().item()),
        "noise_prediction_mse": float((eps_hat - eps).pow(2).mean().item()),
        "root_feature_reconstruction_mse": float((reconstructed[..., 0:6] - original[..., 0:6]).pow(2).mean().item()),
        "joint_feature_reconstruction_mse": float((reconstructed[..., 6:180] - original[..., 6:180]).pow(2).mean().item()),
        "ee_feature_reconstruction_mse": float((reconstructed[..., 180:192] - original[..., 180:192]).pow(2).mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 MuJoCo viewer 中审查 SMP prior 的单步去噪重建效果。")
    parser.add_argument("--checkpoint", required=True, help="SMP prior checkpoint。")
    parser.add_argument("--input", required=True, help="输入 csv 或 npz。")
    parser.add_argument("--mode", default="auto", choices=["auto", "raw", "frames", "features", "continuous", "sample", "window"], help="npz 输入模式。")
    parser.add_argument("--csv-fps", type=float, default=30.0, help="输入为 csv 时使用的 fps。")
    parser.add_argument("--window-size", type=int, default=None, help="窗口长度，默认读取 checkpoint 中的 window_size。")
    parser.add_argument("--stride", type=int, default=1, help="从连续 frames 切窗口时的步长。")
    parser.add_argument("--window-index", type=int, default=0, help="选择第几个窗口。")
    parser.add_argument("--num-windows", type=int, default=25, help="连续取多少个窗口并拼接回放。")
    parser.add_argument("--sample-index", type=int, default=0, help="输入为 samples_denormalized 时选择第几个 sample。")
    parser.add_argument("--timestep", type=int, default=None, help="指定扩散时间步；默认使用 checkpoint 的第一个 reward timestep。")
    parser.add_argument("--style-name", default=None, help="可选：风格名。")
    parser.add_argument("--style-id", type=int, default=None, help="可选：风格 id。")
    parser.add_argument("--device", default="cpu", help="推理设备，例如 cpu 或 cuda:0。")
    parser.add_argument("--seed", type=int, default=0, help="随机种子。")
    parser.add_argument("--urdf", default=None, help="可选：显式指定 G1 URDF。")
    parser.add_argument("--default-height", type=float, default=0.74, help="192D 解码回放时使用的默认 pelvis 高度。")
    parser.add_argument("--playback-fps-scale", type=float, default=1.0, help="播放速度缩放；0.5 表示放慢到一半。")
    parser.add_argument("--hold", type=float, default=0.75, help="每段轨迹播完后停留秒数。")
    parser.add_argument("--loop", action="store_true", help="循环播放。")
    parser.add_argument("--dry-run", action="store_true", help="只打印摘要，不打开 viewer。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prior = _load_prior_bundle(
        checkpoint_path=args.checkpoint,
        device=args.device,
        style_name=args.style_name,
        style_id=args.style_id,
    )

    window_size = int(prior.checkpoint.get("window_size", prior.checkpoint.get("model_cfg", {}).get("window_size", 10)))
    if args.window_size is not None:
        window_size = int(args.window_size)

    timestep = int(prior.timesteps_k[0] if args.timestep is None else args.timestep)
    window_batch, feature_fps, source_mode = _PLAYBACK_UTILS.load_feature_window_batch_from_input(
        input_path=args.input,
        window_size=window_size,
        stride=args.stride,
        mode=args.mode,
        csv_fps=args.csv_fps,
        sample_index=args.sample_index,
        window_index=args.window_index,
        num_windows=args.num_windows,
        urdf_path=args.urdf,
    )

    x0 = torch.as_tensor(window_batch, dtype=torch.float32, device=args.device)
    noisy_bundle = build_noisy_window_bundle(
        x0=x0,
        scheduler=prior.scheduler,
        timestep=timestep,
        seed=args.seed,
        device=args.device,
    )
    eps_hat = _predict_eps(prior, noisy_bundle["xt"], noisy_bundle["t"])
    x0_hat = reconstruct_x0_from_eps(
        xt=noisy_bundle["xt"],
        t=noisy_bundle["t"],
        eps_hat=eps_hat,
        scheduler=prior.scheduler,
    )

    summary = _compute_summary(
        original=noisy_bundle["x0"],
        xt=noisy_bundle["xt"],
        reconstructed=x0_hat,
        eps=noisy_bundle["eps"],
        eps_hat=eps_hat,
    )
    print(f"source_mode: {source_mode}")
    print(f"window_size: {window_size}")
    print(f"num_windows: {args.num_windows}")
    print(f"feature_fps: {feature_fps:.2f}")
    print(f"timestep: {timestep}")
    if timestep >= int(0.8 * prior.scheduler.num_steps):
        print("warning: 当前 timestep 很高，noisy 轨迹会接近纯噪声，乱飞是预期现象。")
    for key, value in summary.items():
        print(f"{key}: {value:.6f}")

    playback_fps = feature_fps * float(args.playback_fps_scale)
    original_features = stitch_overlapping_windows(noisy_bundle["x0"], stride=args.stride).detach().cpu().numpy()
    noisy_features = stitch_overlapping_windows(noisy_bundle["xt"], stride=args.stride).detach().cpu().numpy()
    reconstructed_features = stitch_overlapping_windows(x0_hat, stride=args.stride).detach().cpu().numpy()

    original_traj = _PLAYBACK_UTILS.decode_features_to_trajectory(
        sequence_name=f"{Path(args.input).name}:original",
        features=original_features,
        fps=playback_fps,
        source_mode="original",
        extractor=None,
        default_height=args.default_height,
    )
    noisy_traj = _PLAYBACK_UTILS.decode_features_to_trajectory(
        sequence_name=f"{Path(args.input).name}:noisy_t{timestep}",
        features=noisy_features,
        fps=playback_fps,
        source_mode=f"noisy_t{timestep}",
        extractor=None,
        default_height=args.default_height,
    )
    reconstructed_traj = _PLAYBACK_UTILS.decode_features_to_trajectory(
        sequence_name=f"{Path(args.input).name}:reconstructed",
        features=reconstructed_features,
        fps=playback_fps,
        source_mode=f"reconstructed_from_t{timestep}",
        extractor=None,
        default_height=args.default_height,
    )

    if args.dry_run:
        return

    model = _PLAYBACK_UTILS.build_floating_model_from_urdf(_PLAYBACK_UTILS.resolve_g1_urdf_path(args.urdf))
    _PLAYBACK_UTILS.play_trajectories(
        model=model,
        trajectories=[original_traj, noisy_traj, reconstructed_traj],
        loop=args.loop,
        hold=args.hold,
    )


if __name__ == "__main__":
    main()
