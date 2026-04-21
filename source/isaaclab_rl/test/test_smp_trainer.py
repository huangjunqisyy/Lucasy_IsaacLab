# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib
import sys
from pathlib import Path

import numpy as np
import torch


def _import_trainer_module():
    repo_root = Path(__file__).resolve().parents[3]
    nested_rsl_rl_root = repo_root / "rsl_rl"
    if str(nested_rsl_rl_root) not in sys.path:
        sys.path.insert(0, str(nested_rsl_rl_root))
    return importlib.import_module("rsl_rl.diffusion.trainer")


def _write_toy_motion_dataset(path: Path, num_frames: int = 12, feature_dim: int = 6) -> None:
    frames = np.linspace(-1.0, 1.0, num_frames * feature_dim, dtype=np.float32).reshape(num_frames, feature_dim)
    np.savez(path, fps=np.array([30], dtype=np.int64), frames=frames)


def test_smp_trainer_pretrain_samples_all_diffusion_steps(tmp_path, monkeypatch):
    trainer_module = _import_trainer_module()
    dataset_path = tmp_path / "toy_motion.npz"
    _write_toy_motion_dataset(dataset_path)

    trainer = trainer_module.SMPDiffusionTrainer(
        dataset_path=dataset_path,
        log_dir=tmp_path / "logs",
        batch_size=4,
        max_iters=1,
        window_size=4,
        stride=1,
        num_diffusion_steps=7,
        timesteps_k=[5, 3, 1],
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        device="cpu",
    )

    observed = {}

    def _fake_sample_timesteps(batch_size: int, device=None, timesteps_k=None):
        observed["batch_size"] = batch_size
        observed["timesteps_k"] = timesteps_k
        return torch.tensor([0, 1, 2, 6], device=device, dtype=torch.long)

    monkeypatch.setattr(trainer.scheduler, "sample_timesteps", _fake_sample_timesteps)

    trainer._compute_loss(trainer._next_batch())

    assert observed["batch_size"] == 4
    assert observed["timesteps_k"] is None


def test_smp_trainer_prints_pretrain_progress(tmp_path, monkeypatch, capsys):
    trainer_module = _import_trainer_module()
    dataset_path = tmp_path / "toy_motion.npz"
    _write_toy_motion_dataset(dataset_path)
    monkeypatch.setattr(trainer_module, "log_smp_pretrain_metrics", lambda *args, **kwargs: None)

    trainer = trainer_module.SMPDiffusionTrainer(
        dataset_path=dataset_path,
        log_dir=tmp_path / "logs",
        batch_size=2,
        max_iters=2,
        window_size=4,
        stride=1,
        num_diffusion_steps=7,
        timesteps_k=[5, 3, 1],
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        device="cpu",
    )

    result = trainer.train()
    output = capsys.readouterr().out
    lines = output.splitlines()

    assert result["final_loss"] is not None
    assert "[SMP Pretrain] Starting" in output
    assert "Pretrain timesteps:" in output
    assert "Reward timesteps k:" in output
    assert any("Learning iteration 1/2" in line for line in lines)
    assert any("Learning iteration 2/2" in line for line in lines)
    assert any("Computation:" in line for line in lines)
    assert any("Mean loss:" in line for line in lines)
    assert any("Mean loss (running):" in line for line in lines)
    assert any("Estimated total time:" in line for line in lines)
    assert any("Time remaining:" in line for line in lines)
    assert any("Checkpoint:" in line for line in lines)


def test_smp_trainer_uses_tqdm_progress_bar(tmp_path, monkeypatch):
    trainer_module = _import_trainer_module()
    dataset_path = tmp_path / "toy_motion.npz"
    _write_toy_motion_dataset(dataset_path)
    monkeypatch.setattr(trainer_module, "log_smp_pretrain_metrics", lambda *args, **kwargs: None)

    observed = {}

    class _FakeTqdm:
        def __init__(self, iterable, total=None, desc=None, leave=None, dynamic_ncols=None):
            self._iterable = list(iterable)
            self.total = total
            self.desc = desc
            self.leave = leave
            self.dynamic_ncols = dynamic_ncols
            self.steps = 0
            self.closed = False
            self.postfix_history = []
            observed["instance"] = self

        def __iter__(self):
            for item in self._iterable:
                self.steps += 1
                yield item

        def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
            postfix = dict(ordered_dict or {})
            postfix.update(kwargs)
            self.postfix_history.append(postfix)

        def close(self):
            self.closed = True

    monkeypatch.setattr(trainer_module, "tqdm", _FakeTqdm, raising=False)

    trainer = trainer_module.SMPDiffusionTrainer(
        dataset_path=dataset_path,
        log_dir=tmp_path / "logs",
        batch_size=2,
        max_iters=2,
        window_size=4,
        stride=1,
        num_diffusion_steps=7,
        timesteps_k=[5, 3, 1],
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        device="cpu",
    )

    result = trainer.train()

    bar = observed["instance"]
    assert result["final_loss"] is not None
    assert bar.total == 2
    assert bar.desc == "SMP pretrain"
    assert bar.dynamic_ncols is True
    assert bar.steps == 2
    assert bar.closed is True
    assert any("loss" in postfix for postfix in bar.postfix_history)
    assert any("running_loss" in postfix for postfix in bar.postfix_history)
