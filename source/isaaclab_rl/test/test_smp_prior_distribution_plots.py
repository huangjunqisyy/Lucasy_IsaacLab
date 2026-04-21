import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_plot_module():
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "scripts" / "imitation_learning" / "smp" / "plot_smp_prior_distribution.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_prior_distribution_plots_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find scripts/imitation_learning/smp/plot_smp_prior_distribution.py")


def test_extract_named_series_splits_g1_192d_blocks():
    module = _load_plot_module()
    windows = np.zeros((2, 4, 192), dtype=np.float32)
    windows[..., 0] = 1.0
    windows[..., 3] = 2.0
    windows[..., 180 + 2] = 0.3
    windows[..., 183 + 2] = 0.4

    series = module.extract_named_series(windows)

    assert "base_lin_vel_b/x" in series
    assert "base_ang_vel_b/x" in series
    assert "ee_pos_b/left_ankle_roll_link/z" in series
    assert "ee_pos_b/right_ankle_roll_link/z" in series
    assert "norm/base_lin_vel_b" in series
    assert "temporal/diff_l2" in series
    assert series["base_lin_vel_b/x"].shape == (8,)
    assert np.allclose(series["base_lin_vel_b/x"], 1.0)
    assert np.allclose(series["base_ang_vel_b/x"], 2.0)
    assert np.allclose(series["ee_pos_b/left_ankle_roll_link/z"], 0.3)
    assert np.allclose(series["ee_pos_b/right_ankle_roll_link/z"], 0.4)


def test_build_series_report_marks_generated_ratio():
    module = _load_plot_module()
    reference = {
        "base_lin_vel_b/x": np.array([0.0, 1.0, 2.0], dtype=np.float32),
        "temporal/diff_l2": np.array([1.0, 1.0, 1.0], dtype=np.float32),
    }
    generated = {
        "base_lin_vel_b/x": np.array([0.0, 0.5, 1.0], dtype=np.float32),
        "temporal/diff_l2": np.array([0.5, 0.5, 0.5], dtype=np.float32),
    }

    rows = module.build_series_report(reference, generated)
    by_name = {row["name"]: row for row in rows}

    assert by_name["base_lin_vel_b/x"]["generated_std_ratio"] < 1.0
    assert by_name["temporal/diff_l2"]["generated_mean_ratio"] == 0.5
    assert set(rows[0]) >= {
        "name",
        "reference_mean",
        "generated_mean",
        "generated_mean_ratio",
        "reference_std",
        "generated_std",
        "generated_std_ratio",
    }


def test_select_plot_series_prefers_base_and_ee_keys():
    module = _load_plot_module()
    keys = [
        "joint_rot6d_rel/dim_000",
        "base_lin_vel_b/x",
        "base_ang_vel_b/z",
        "ee_pos_b/left_ankle_roll_link/z",
        "temporal/diff_l2",
    ]

    selected = module.select_plot_series(keys)

    assert selected[:4] == [
        "base_lin_vel_b/x",
        "base_ang_vel_b/z",
        "ee_pos_b/left_ankle_roll_link/z",
        "temporal/diff_l2",
    ]
