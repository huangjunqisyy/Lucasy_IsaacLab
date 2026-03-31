# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _cfg_get(container, key: str, default=None):
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    return getattr(container, key, default)


def _cfg_to_dict(container):
    if container is None:
        return None
    if isinstance(container, dict):
        return {key: _cfg_to_dict(value) for key, value in container.items()}
    if hasattr(container, "to_dict"):
        return _cfg_to_dict(container.to_dict())
    return container


def _unwrap_env(env):
    return getattr(env, "unwrapped", env)


def _normalize_env_ids(env, env_ids, asset_name: str) -> torch.Tensor:
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(dtype=torch.long)
    if env_ids is not None:
        return torch.as_tensor(env_ids, dtype=torch.long)

    target_env = _unwrap_env(env)
    asset = target_env.scene[asset_name]
    device = asset.data.default_root_state.device
    return torch.arange(asset.data.default_root_state.shape[0], device=device, dtype=torch.long)


@dataclass
class SMPPreviewRuntime:
    """保存 GSI 预览所需的冻结 prior 与采样配置。"""

    gsi_sampler: Any
    style_program: dict[str, object]
    guidance_scale: float
    max_resample_attempts: int
    fallback_to_default_reset: bool
    asset_name: str = "robot"


@dataclass
class SMPPreviewResetResult:
    """记录一次手动 GSI reset 触发的结果，便于脚本打印与测试断言。"""

    decode_result: Any | None
    applied: bool
    fallback_to_default: bool
    sample_attempts: int
    env_ids: torch.Tensor


def build_smp_preview_runtime(agent_cfg, device: str | torch.device = "cpu") -> SMPPreviewRuntime:
    """复用 SMP runner 的 prior/style/GSI 构建逻辑，避免在预览脚本里复制实现。"""
    from rsl_rl.runners import SMPOnPolicyRunner

    preview_runner = object.__new__(SMPOnPolicyRunner)
    preview_runner.device = torch.device(device)
    preview_runner.smp_prior_cfg = _cfg_to_dict(_cfg_get(agent_cfg, "smp_prior", None))
    if preview_runner.smp_prior_cfg is None:
        raise ValueError("Preview requires agent_cfg.smp_prior")
    preview_runner.style_cfg = _cfg_to_dict(_cfg_get(preview_runner.smp_prior_cfg, "style_cfg", None))
    preview_runner.gsi_cfg = _cfg_to_dict(_cfg_get(agent_cfg, "gsi_cfg", None))
    if preview_runner.gsi_cfg is None:
        raise ValueError("Preview requires agent_cfg.gsi_cfg")

    preview_runner.smp_prior = preview_runner._load_prior_model()
    preview_runner.style_program = preview_runner._resolve_style_program_from_cfg()
    preview_runner.gsi_sampler = preview_runner._build_gsi_sampler()
    if preview_runner.gsi_sampler is None:
        raise ValueError("GSI preview requires gsi_cfg.enabled=True and valid feature block offsets")

    guidance_scale = _cfg_get(preview_runner.gsi_cfg, "guidance_scale", None)
    if guidance_scale is None:
        guidance_scale = float(preview_runner.style_program.get("guidance_scale", 1.0))

    return SMPPreviewRuntime(
        gsi_sampler=preview_runner.gsi_sampler,
        style_program=dict(preview_runner.style_program),
        guidance_scale=float(guidance_scale),
        max_resample_attempts=max(1, int(_cfg_get(preview_runner.gsi_cfg, "max_resample_attempts", 1))),
        fallback_to_default_reset=bool(_cfg_get(preview_runner.gsi_cfg, "fallback_to_default_reset", True)),
        asset_name=str(_cfg_get(preview_runner.gsi_cfg, "asset_name", "robot")),
    )


def sample_and_apply_preview_reset(env, runtime: SMPPreviewRuntime, env_ids=None) -> SMPPreviewResetResult:
    """对指定环境触发一次手动 GSI reset，并按 runner 语义决定是否真正写回仿真。"""
    from isaaclab_tasks.manager_based.locomotion.velocity.mdp.smp_reset import (
        apply_smp_reset_state,
        build_smp_reset_reference,
    )

    resolved_env_ids = _normalize_env_ids(env, env_ids, asset_name=runtime.asset_name)
    reference_state = build_smp_reset_reference(env, env_ids=resolved_env_ids, asset_name=runtime.asset_name)

    last_result = None
    for attempt_idx in range(runtime.max_resample_attempts):
        last_result = runtime.gsi_sampler.sample_reset_state(
            batch_size=int(resolved_env_ids.numel()),
            reference_state=reference_state,
            style_program=runtime.style_program,
            guidance_scale=float(runtime.guidance_scale),
        )
        if last_result.supports_reset_state:
            apply_smp_reset_state(env, env_ids=resolved_env_ids, state=last_result.state, asset_name=runtime.asset_name)
            return SMPPreviewResetResult(
                decode_result=last_result,
                applied=True,
                fallback_to_default=False,
                sample_attempts=attempt_idx + 1,
                env_ids=resolved_env_ids.clone(),
            )

    if last_result is not None and not runtime.fallback_to_default_reset:
        apply_smp_reset_state(env, env_ids=resolved_env_ids, state=last_result.state, asset_name=runtime.asset_name)
        return SMPPreviewResetResult(
            decode_result=last_result,
            applied=True,
            fallback_to_default=False,
            sample_attempts=runtime.max_resample_attempts,
            env_ids=resolved_env_ids.clone(),
        )

    return SMPPreviewResetResult(
        decode_result=last_result,
        applied=False,
        fallback_to_default=True,
        sample_attempts=runtime.max_resample_attempts,
        env_ids=resolved_env_ids.clone(),
    )


def _select_row(tensor: torch.Tensor, row_index: int) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.detach().cpu()
    return tensor[row_index].detach().cpu()


def format_preview_reset_summary(reset_result: SMPPreviewResetResult, row_index: int = 0) -> str:
    """把一次采样结果压缩成终端可读摘要，便于观察当前 reset state。"""
    if reset_result.decode_result is None:
        return (
            f"[SMP GSI Preview] env_ids={reset_result.env_ids.tolist()} applied=no "
            f"fallback_to_default={str(reset_result.fallback_to_default).lower()} no-sample"
        )

    decode_result = reset_result.decode_result
    state = decode_result.state
    root_pos = _select_row(state.root_pos_w, row_index)
    root_lin_vel = _select_row(state.root_lin_vel_w, row_index)
    root_ang_vel = _select_row(state.root_ang_vel_w, row_index)
    joint_pos = _select_row(state.joint_pos, row_index)
    joint_vel = _select_row(state.joint_vel, row_index)
    unrecoverable = list(getattr(decode_result, "unrecoverable_feature_blocks", ()))

    return (
        f"[SMP GSI Preview] env_ids={reset_result.env_ids.tolist()} "
        f"applied={str(reset_result.applied).lower()} "
        f"fallback_to_default={str(reset_result.fallback_to_default).lower()} "
        f"supports_reset_state={str(bool(decode_result.supports_reset_state)).lower()} "
        f"attempts={reset_result.sample_attempts} "
        f"mse={float(decode_result.reconstruction_mse):.6f} "
        f"unrecoverable={unrecoverable if unrecoverable else ['none']} "
        f"root_pos={[round(float(value), 4) for value in root_pos.tolist()]} "
        f"root_lin_vel_norm={float(root_lin_vel.norm().item()):.4f} "
        f"root_ang_vel_norm={float(root_ang_vel.norm().item()):.4f} "
        f"joint_pos[min,max]=({float(joint_pos.min().item()):.4f}, {float(joint_pos.max().item()):.4f}) "
        f"joint_vel_norm={float(joint_vel.norm().item()):.4f}"
    )
