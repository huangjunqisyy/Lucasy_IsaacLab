import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from rsl_rl.diffusion.scheduler import DiffusionScheduler


def _load_denoising_module():
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "scripts" / "imitation_learning" / "smp" / "inspect_smp_denoising_in_mujoco.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_denoising_viewer_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find scripts/imitation_learning/smp/inspect_smp_denoising_in_mujoco.py")


def test_reconstruct_x0_from_eps_inverts_q_sample():
    module = _load_denoising_module()
    scheduler = DiffusionScheduler(num_steps=10)

    x0 = torch.randn(2, 10, 192)
    eps = torch.randn_like(x0)
    t = torch.tensor([2, 7], dtype=torch.long)
    xt = scheduler.q_sample(x0, t, eps)

    x0_hat = module.reconstruct_x0_from_eps(xt=xt, t=t, eps_hat=eps, scheduler=scheduler)

    assert torch.allclose(x0_hat, x0, atol=1e-5)


def test_build_noisy_window_bundle_preserves_shapes():
    module = _load_denoising_module()
    scheduler = DiffusionScheduler(num_steps=12)

    x0 = torch.zeros(1, 10, 192)
    bundle = module.build_noisy_window_bundle(
        x0=x0,
        scheduler=scheduler,
        timestep=5,
        seed=7,
        device="cpu",
    )

    assert bundle["x0"].shape == (1, 10, 192)
    assert bundle["xt"].shape == (1, 10, 192)
    assert bundle["eps"].shape == (1, 10, 192)
    assert bundle["t"].shape == (1,)
    assert int(bundle["t"][0].item()) == 5


def test_stitch_overlapping_windows_recovers_original_sequence():
    module = _load_denoising_module()

    sequence = torch.arange(5 * 3, dtype=torch.float32).view(5, 3)
    windows = torch.stack([sequence[0:3], sequence[1:4], sequence[2:5]], dim=0)

    stitched = module.stitch_overlapping_windows(windows, stride=1)

    assert torch.allclose(stitched, sequence)


def test_select_windows_accepts_pre_windowed_batches():
    module = _load_denoising_module()

    windows = np.zeros((4, 10, 192), dtype=np.float32)

    selected = module._select_windows(windows, window_size=10, stride=1, window_index=0, num_windows=4)

    assert selected.shape == (4, 10, 192)
