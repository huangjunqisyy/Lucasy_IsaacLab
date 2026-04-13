# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NESTED_RSL_RL_ROOT = _REPO_ROOT / "rsl_rl"
if str(_NESTED_RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(_NESTED_RSL_RL_ROOT))

from rsl_rl.diffusion.model import MotionEpsilonTransformer
from rsl_rl.diffusion.scheduler import DiffusionScheduler


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXPORT_DATASET_MODULE = _load_module(
    "isaaclab_smp_diagnostics_export_dataset",
    _REPO_ROOT / "scripts" / "imitation_learning" / "smp" / "export_g1_motion_dataset.py",
)


@dataclass
class GroupScoreResult:
    label: str
    num_windows: int
    raw_noise_mse: np.ndarray
    per_timestep_raw_mse: dict[int, np.ndarray]
    source_paths: list[str]


@dataclass
class GroupScoreSummary:
    label: str
    num_windows: int
    raw_noise_mse_mean: float
    raw_noise_mse_std: float
    normalized_noise_mse_mean: float
    normalized_noise_mse_std: float
    reward_mean: float
    reward_std: float
    reward_p10: float
    reward_p50: float
    reward_p90: float
    per_timestep_raw_mse_mean: dict[int, float]
    source_paths: list[str]


class SMPWindowCollector:
    """Collect flattened `smp_motion_window` observations and save them as a policy dataset."""

    def __init__(self, obs_group: str, window_size: int, feature_dim: int):
        self.obs_group = str(obs_group)
        self.window_size = int(window_size)
        self.feature_dim = int(feature_dim)
        self._windows: list[np.ndarray] = []
        self._step_indices: list[np.ndarray] = []
        self._env_ids: list[np.ndarray] = []
        self._steps_recorded = 0

    def add(self, obs: dict[str, torch.Tensor]) -> None:
        if self.obs_group not in obs:
            raise KeyError(f"SMP observation group '{self.obs_group}' not found in observations")
        smp_window = obs[self.obs_group]
        if smp_window.ndim != 2 or smp_window.shape[-1] != self.window_size * self.feature_dim:
            raise ValueError(
                f"Expected flattened SMP window shape (B, {self.window_size * self.feature_dim}), got {tuple(smp_window.shape)}"
            )

        windows = smp_window.detach().float().cpu().view(smp_window.shape[0], self.window_size, self.feature_dim).numpy()
        self._windows.append(windows)
        self._step_indices.append(np.full((smp_window.shape[0],), self._steps_recorded, dtype=np.int64))
        self._env_ids.append(np.arange(smp_window.shape[0], dtype=np.int64))
        self._steps_recorded += 1

    @property
    def num_steps_recorded(self) -> int:
        return self._steps_recorded

    def save(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._windows:
            raise ValueError("No SMP windows were collected")

        payload = {
            "windows": np.concatenate(self._windows, axis=0).astype(np.float32),
            "step_index": np.concatenate(self._step_indices, axis=0),
            "env_id": np.concatenate(self._env_ids, axis=0),
            "window_size": np.array([self.window_size], dtype=np.int64),
            "feature_dim": np.array([self.feature_dim], dtype=np.int64),
            "obs_group": np.asarray([self.obs_group], dtype=np.str_),
            "num_steps_recorded": np.array([self._steps_recorded], dtype=np.int64),
        }
        np.savez(output_path, **payload)
        return output_path


def build_window_collector_from_observation(
    obs: dict[str, torch.Tensor],
    *,
    obs_group: str,
    window_size: int,
) -> SMPWindowCollector:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if obs_group not in obs:
        raise KeyError(f"SMP observation group '{obs_group}' not found in observations")

    smp_window = obs[obs_group]
    if smp_window.ndim != 2:
        raise ValueError(f"Expected flattened SMP window shape (B, W*F), got {tuple(smp_window.shape)}")
    flattened_dim = int(smp_window.shape[-1])
    if flattened_dim % int(window_size) != 0:
        raise ValueError(
            f"Could not infer feature_dim from flattened SMP window dim {flattened_dim} and window_size {window_size}"
        )
    feature_dim = flattened_dim // int(window_size)
    return SMPWindowCollector(obs_group=str(obs_group), window_size=int(window_size), feature_dim=feature_dim)


def build_sliding_windows(frames: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    if frames.ndim != 2:
        raise ValueError(f"Expected frames to have shape (num_frames, feature_dim), got {tuple(frames.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if frames.shape[0] < window_size:
        raise ValueError(f"Expected at least {window_size} frames, got {frames.shape[0]}")
    windows = [frames[start : start + window_size] for start in range(0, frames.shape[0] - window_size + 1, stride)]
    return np.stack(windows, axis=0).astype(np.float32)


def _convert_raw_motion_to_frames(input_path: Path) -> tuple[np.ndarray, float]:
    raw_motion, fps = _EXPORT_DATASET_MODULE._load_motion_arrays(input_path)
    if "frames" in raw_motion:
        frames = raw_motion["frames"]
    else:
        frames = _EXPORT_DATASET_MODULE._build_smp_frames_from_raw_motion(raw_motion).cpu().numpy()
    return np.asarray(frames, dtype=np.float32), float(fps)


def _save_converted_frames(output_path: Path, frames: np.ndarray, fps: float, window_size: int, stride: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "frames": frames.astype(np.float32),
        "fps": np.array([fps], dtype=np.float32),
        "window_size": np.array([window_size], dtype=np.int64),
        "stride": np.array([stride], dtype=np.int64),
        "feature_dim": np.array([frames.shape[-1]], dtype=np.int64),
        "joint_names": np.asarray(_EXPORT_DATASET_MODULE.g1_smp_joint_names),
        "joint_axes": np.asarray(_EXPORT_DATASET_MODULE.g1_smp_joint_axes, dtype=np.float32),
        "ee_names": np.asarray(_EXPORT_DATASET_MODULE.g1_ee_names),
        "source_name": np.asarray([output_path.stem], dtype=np.str_),
    }
    np.savez(output_path, **payload)
    return output_path


def load_motion_windows(
    input_path: str | Path,
    *,
    dataset_label: str,
    window_size: int,
    stride: int = 1,
    converted_dir: str | Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    input_path = Path(input_path)
    with np.load(input_path) as data:
        keys = set(data.files)
        if "windows" in keys:
            windows = np.asarray(data["windows"], dtype=np.float32)
            if windows.ndim == 4:
                windows = windows.reshape(-1, windows.shape[-2], windows.shape[-1])
            if windows.ndim != 3:
                raise ValueError(f"Expected policy windows to have ndim 3 or 4, got {windows.ndim}")
            return windows, {
                "kind": "policy_windows",
                "source_path": str(input_path),
                "converted_path": None,
            }

    frames, fps = _convert_raw_motion_to_frames(input_path)
    converted_path = None
    if converted_dir is not None:
        converted_path = _save_converted_frames(
            Path(converted_dir) / f"{dataset_label}_{input_path.stem}.npz",
            frames=frames,
            fps=fps,
            window_size=window_size,
            stride=stride,
        )
    windows = build_sliding_windows(frames, window_size=window_size, stride=stride)
    return windows, {
        "kind": "frames_dataset",
        "source_path": str(input_path),
        "converted_path": None if converted_path is None else str(converted_path),
    }


@dataclass
class SMPPriorBundle:
    model: MotionEpsilonTransformer
    scheduler: DiffusionScheduler
    timesteps_k: list[int]
    reward_scale: float
    style_id: int | None
    style_name: str | None
    checkpoint: dict[str, object]


def _resolve_style_id(checkpoint: dict[str, object], style_name: str | None = None, style_id: int | None = None) -> tuple[int | None, str | None]:
    model_cfg = checkpoint.get("model_cfg", {})
    num_styles = int(model_cfg.get("num_styles", 0))
    if num_styles <= 0:
        return None, None

    style_cfg = checkpoint.get("style_cfg", {})
    style_to_id = dict(style_cfg.get("style_to_id", {}))
    if style_id is not None:
        resolved_name = None
        for candidate_name, candidate_id in style_to_id.items():
            if int(candidate_id) == int(style_id):
                resolved_name = str(candidate_name)
                break
        return int(style_id), resolved_name

    if style_name is not None:
        if style_name not in style_to_id:
            raise KeyError(f"Unknown style name '{style_name}' in checkpoint")
        return int(style_to_id[style_name]), str(style_name)

    if len(style_to_id) == 1:
        only_name, only_id = next(iter(style_to_id.items()))
        return int(only_id), str(only_name)
    raise ValueError("Checkpoint contains multiple styles; provide style_name or style_id explicitly")


def load_prior_bundle(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    style_name: str | None = None,
    style_id: int | None = None,
    timesteps_k: list[int] | None = None,
    reward_scale: float | None = None,
) -> SMPPriorBundle:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = checkpoint.get("model_cfg", {})
    model = MotionEpsilonTransformer(
        feature_dim=int(model_cfg["feature_dim"]),
        window_size=int(model_cfg["window_size"]),
        num_diffusion_steps=int(model_cfg["num_diffusion_steps"]),
        num_styles=int(model_cfg.get("num_styles", 0)),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    resolved_style_id, resolved_style_name = _resolve_style_id(checkpoint, style_name=style_name, style_id=style_id)
    resolved_timesteps_k = list(checkpoint.get("timesteps_k", [])) if timesteps_k is None else list(timesteps_k)
    if len(resolved_timesteps_k) == 0:
        raise ValueError("timesteps_k must be available in checkpoint or provided explicitly")

    return SMPPriorBundle(
        model=model,
        scheduler=DiffusionScheduler(num_steps=int(model_cfg["num_diffusion_steps"])),
        timesteps_k=[int(timestep) for timestep in resolved_timesteps_k],
        reward_scale=float(1.0 if reward_scale is None else reward_scale),
        style_id=resolved_style_id,
        style_name=resolved_style_name,
        checkpoint=checkpoint,
    )


def score_motion_windows(
    windows: np.ndarray,
    *,
    prior: SMPPriorBundle,
    batch_size: int = 128,
    noise_draws: int = 4,
    device: str | torch.device = "cpu",
    seed: int = 0,
    label: str,
    source_paths: list[str],
) -> GroupScoreResult:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shape (N, W, F), got {tuple(windows.shape)}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if noise_draws <= 0:
        raise ValueError(f"noise_draws must be positive, got {noise_draws}")

    device = torch.device(device)
    tensor_windows = torch.as_tensor(windows, dtype=torch.float32, device=device)
    style_id_tensor = None
    if prior.style_id is not None:
        style_id_tensor = torch.full((batch_size,), int(prior.style_id), device=device, dtype=torch.long)

    generator = torch.Generator(device=device.type if device.type != "mps" else "cpu")
    generator.manual_seed(int(seed))
    per_timestep_chunks = {int(timestep): [] for timestep in prior.timesteps_k}
    raw_noise_mse_chunks = []

    with torch.inference_mode():
        for start in range(0, tensor_windows.shape[0], batch_size):
            batch = tensor_windows[start : start + batch_size]
            batch_style_id = style_id_tensor[: batch.shape[0]] if style_id_tensor is not None else None
            batch_per_timestep = {}
            for timestep in prior.timesteps_k:
                timestep_draws = []
                t = torch.full((batch.shape[0],), int(timestep), device=device, dtype=torch.long)
                for _ in range(noise_draws):
                    eps = torch.randn(batch.shape, generator=generator, device=device, dtype=batch.dtype)
                    xt = prior.scheduler.q_sample(batch, t, eps)
                    eps_hat = prior.model(xt, t, style_id=batch_style_id)
                    mse = (eps_hat - eps).pow(2).flatten(start_dim=1).mean(dim=1)
                    timestep_draws.append(mse)
                batch_per_timestep[int(timestep)] = torch.stack(timestep_draws, dim=0).mean(dim=0)
                per_timestep_chunks[int(timestep)].append(batch_per_timestep[int(timestep)].cpu().numpy())
            raw_noise_mse = torch.stack([batch_per_timestep[int(timestep)] for timestep in prior.timesteps_k], dim=0).mean(dim=0)
            raw_noise_mse_chunks.append(raw_noise_mse.cpu().numpy())

    per_timestep_raw_mse = {
        int(timestep): np.concatenate(chunks, axis=0).astype(np.float32)
        for timestep, chunks in per_timestep_chunks.items()
    }
    return GroupScoreResult(
        label=label,
        num_windows=int(windows.shape[0]),
        raw_noise_mse=np.concatenate(raw_noise_mse_chunks, axis=0).astype(np.float32),
        per_timestep_raw_mse=per_timestep_raw_mse,
        source_paths=list(source_paths),
    )


def normalize_and_summarize_group_scores(
    group_scores: list[GroupScoreResult],
    *,
    reward_scale: float,
    reference_label: str = "positive",
) -> dict[str, GroupScoreSummary]:
    if not group_scores:
        raise ValueError("group_scores must be non-empty")
    reference = next((group for group in group_scores if group.label == reference_label), group_scores[0])
    timestep_scale = {
        int(timestep): max(float(values.mean()), 1.0e-6)
        for timestep, values in reference.per_timestep_raw_mse.items()
    }

    summaries: dict[str, GroupScoreSummary] = {}
    for group in group_scores:
        normalized_terms = [
            group.per_timestep_raw_mse[int(timestep)] / timestep_scale[int(timestep)]
            for timestep in sorted(timestep_scale.keys())
        ]
        normalized_noise_mse = np.stack(normalized_terms, axis=0).mean(axis=0).astype(np.float32)
        reward = np.exp(-float(reward_scale) * normalized_noise_mse).astype(np.float32)
        summaries[group.label] = GroupScoreSummary(
            label=group.label,
            num_windows=group.num_windows,
            raw_noise_mse_mean=float(group.raw_noise_mse.mean()),
            raw_noise_mse_std=float(group.raw_noise_mse.std()),
            normalized_noise_mse_mean=float(normalized_noise_mse.mean()),
            normalized_noise_mse_std=float(normalized_noise_mse.std()),
            reward_mean=float(reward.mean()),
            reward_std=float(reward.std()),
            reward_p10=float(np.percentile(reward, 10)),
            reward_p50=float(np.percentile(reward, 50)),
            reward_p90=float(np.percentile(reward, 90)),
            per_timestep_raw_mse_mean={
                int(timestep): float(values.mean()) for timestep, values in sorted(group.per_timestep_raw_mse.items())
            },
            source_paths=list(group.source_paths),
        )
    return summaries


def format_group_summary_table(group_summaries: dict[str, GroupScoreSummary]) -> str:
    headers = [
        "group",
        "windows",
        "raw_mse_mean",
        "raw_mse_std",
        "norm_mse_mean",
        "reward_mean",
        "reward_p50",
        "reward_p90",
    ]
    lines = [" | ".join(headers), " | ".join("-" * len(header) for header in headers)]
    for label, summary in group_summaries.items():
        lines.append(
            " | ".join(
                [
                    label,
                    str(summary.num_windows),
                    f"{summary.raw_noise_mse_mean:.6f}",
                    f"{summary.raw_noise_mse_std:.6f}",
                    f"{summary.normalized_noise_mse_mean:.6f}",
                    f"{summary.reward_mean:.6f}",
                    f"{summary.reward_p50:.6f}",
                    f"{summary.reward_p90:.6f}",
                ]
            )
        )
    return "\n".join(lines)


def format_group_detail_lines(group_summaries: dict[str, GroupScoreSummary]) -> str:
    lines: list[str] = []
    for label, summary in group_summaries.items():
        lines.append(f"[{label}]")
        lines.append(f"  windows={summary.num_windows}")
        lines.append(f"  reward_mean={summary.reward_mean:.6f} reward_std={summary.reward_std:.6f}")
        lines.append(
            f"  raw_noise_mse_mean={summary.raw_noise_mse_mean:.6f} normalized_noise_mse_mean={summary.normalized_noise_mse_mean:.6f}"
        )
        lines.append(
            "  per_timestep_raw_mse="
            + ", ".join(f"t{timestep}={value:.6f}" for timestep, value in summary.per_timestep_raw_mse_mean.items())
        )
        lines.append(f"  sources={summary.source_paths}")
    return "\n".join(lines)
