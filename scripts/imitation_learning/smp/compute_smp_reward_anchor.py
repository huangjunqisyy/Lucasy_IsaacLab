from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
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
    "isaaclab_smp_path_bootstrap_anchor",
    Path(__file__).resolve().with_name("path_bootstrap.py"),
)
_PATH_BOOTSTRAP.prepend_local_source_paths(__file__, package_names=("isaaclab", "isaaclab_rl", "isaaclab_tasks"))
_REPO_ROOT = _PATH_BOOTSTRAP.find_repo_root(__file__)
_SMP_DIAGNOSTICS = _load_module(
    "isaaclab_smp_reward_anchor_diagnostics",
    _REPO_ROOT / "source" / "isaaclab_rl" / "isaaclab_rl" / "rsl_rl" / "smp_diagnostics.py",
)
load_motion_windows = _SMP_DIAGNOSTICS.load_motion_windows
load_prior_bundle = _SMP_DIAGNOSTICS.load_prior_bundle
score_motion_windows = _SMP_DIAGNOSTICS.score_motion_windows


def _expand_reference_paths(reference_inputs: list[str | Path]) -> list[str]:
    resolved_paths: list[str] = []
    for input_path in reference_inputs:
        path = Path(input_path).expanduser().resolve()
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "datasets" in payload:
                for entry in payload["datasets"]:
                    if not isinstance(entry, dict) or "path" not in entry:
                        raise ValueError(f"Invalid dataset manifest entry in {path}: {entry}")
                    resolved_paths.append(str(Path(entry["path"]).expanduser().resolve()))
                continue
        resolved_paths.append(str(path))
    deduped: list[str] = []
    seen = set()
    for path in resolved_paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    if not deduped:
        raise ValueError("reference paths must be non-empty")
    return deduped


def compute_smp_reward_anchor(
    *,
    checkpoint_path: str | Path,
    reference_paths: list[str | Path],
    output_json: str | Path,
    batch_size: int = 128,
    noise_draws: int = 4,
    device: str = "cpu",
    stride: int = 1,
    converted_dir: str | Path | None = None,
    max_windows: int | None = None,
    seed: int = 0,
    style_name: str | None = None,
    style_id: int | None = None,
    timesteps_k: list[int] | None = None,
    reward_scale: float | None = None,
) -> dict[str, object]:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()
    expanded_reference_paths = _expand_reference_paths(reference_paths)

    prior = load_prior_bundle(
        checkpoint_path,
        device=device,
        style_name=style_name,
        style_id=style_id,
        timesteps_k=timesteps_k,
        reward_scale=reward_scale,
    )
    window_size = int(prior.checkpoint["model_cfg"]["window_size"])

    window_batches = []
    source_paths = []
    for input_path in expanded_reference_paths:
        windows, meta = load_motion_windows(
            input_path,
            dataset_label="anchor",
            window_size=window_size,
            stride=stride,
            converted_dir=converted_dir,
        )
        window_batches.append(windows)
        source_paths.append(str(meta["source_path"]))

    all_windows = np.concatenate(window_batches, axis=0).astype(np.float32)
    if max_windows is not None and all_windows.shape[0] > int(max_windows):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(all_windows.shape[0], size=int(max_windows), replace=False))
        all_windows = all_windows[indices]

    result = score_motion_windows(
        all_windows,
        prior=prior,
        batch_size=batch_size,
        noise_draws=noise_draws,
        device=device,
        seed=seed,
        label="reference",
        source_paths=source_paths,
    )

    mu_ref = {
        str(int(timestep)): float(values.mean())
        for timestep, values in sorted(result.per_timestep_raw_mse.items())
    }
    std_ref = {
        str(int(timestep)): float(values.std())
        for timestep, values in sorted(result.per_timestep_raw_mse.items())
    }
    sem_ref = {
        str(int(timestep)): float(values.std() / math.sqrt(max(1, values.shape[0])))
        for timestep, values in sorted(result.per_timestep_raw_mse.items())
    }

    payload = {
        "checkpoint_path": str(checkpoint_path),
        "reference_paths": expanded_reference_paths,
        "window_size": window_size,
        "feature_dim": int(prior.checkpoint["model_cfg"]["feature_dim"]),
        "timesteps_k": [int(timestep) for timestep in prior.timesteps_k],
        "reward_scale": float(prior.reward_scale),
        "style_name": prior.style_name,
        "style_id": prior.style_id,
        "batch_size": int(batch_size),
        "noise_draws": int(noise_draws),
        "stride": int(stride),
        "seed": int(seed),
        "num_windows": int(result.num_windows),
        "raw_noise_mse_mean": float(result.raw_noise_mse.mean()),
        "raw_noise_mse_std": float(result.raw_noise_mse.std()),
        "mu_ref": mu_ref,
        "std_ref": std_ref,
        "sem_ref": sem_ref,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _format_anchor_report(payload: dict[str, object]) -> str:
    lines = [
        f"checkpoint_path: {payload['checkpoint_path']}",
        f"timesteps_k: {payload['timesteps_k']}",
        f"window_size: {payload['window_size']}",
        f"feature_dim: {payload['feature_dim']}",
        f"num_windows: {payload['num_windows']}",
        f"noise_draws: {payload['noise_draws']}",
        f"raw_noise_mse_mean: {payload['raw_noise_mse_mean']:.6f}",
        f"raw_noise_mse_std: {payload['raw_noise_mse_std']:.6f}",
        "",
        "[mu_ref]",
    ]
    for timestep in payload["timesteps_k"]:
        key = str(int(timestep))
        lines.append(
            f"  t{int(timestep)}: mu={payload['mu_ref'][key]:.6f} std={payload['std_ref'][key]:.6f} sem={payload['sem_ref'][key]:.6f}"
        )
    lines.extend(["", f"output_json: {payload['_output_json']}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute fixed SMP normalization anchors from a reference dataset.")
    parser.add_argument("--checkpoint", required=True, help="SMP prior checkpoint，例如 model_latest.pt")
    parser.add_argument("--reference", nargs="+", required=True, help="参考数据路径，可传 manifest.json 或多个 npz/csv")
    parser.add_argument("--output-json", required=True, help="输出 anchor JSON 路径")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-draws", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--converted-dir", default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--style-name", default=None)
    parser.add_argument("--style-id", type=int, default=None)
    parser.add_argument("--timesteps-k", type=int, nargs="*", default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    args = parser.parse_args()

    payload = compute_smp_reward_anchor(
        checkpoint_path=args.checkpoint,
        reference_paths=args.reference,
        output_json=args.output_json,
        batch_size=args.batch_size,
        noise_draws=args.noise_draws,
        device=args.device,
        stride=args.stride,
        converted_dir=args.converted_dir,
        max_windows=args.max_windows,
        seed=args.seed,
        style_name=args.style_name,
        style_id=args.style_id,
        timesteps_k=args.timesteps_k,
        reward_scale=args.reward_scale,
    )
    payload["_output_json"] = str(Path(args.output_json).expanduser().resolve())
    print(_format_anchor_report(payload))


if __name__ == "__main__":
    main()
