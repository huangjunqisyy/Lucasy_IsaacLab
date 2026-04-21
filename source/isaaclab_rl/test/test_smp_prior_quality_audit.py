import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_audit_module():
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "scripts" / "imitation_learning" / "smp" / "audit_smp_prior_quality.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_prior_quality_audit_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find scripts/imitation_learning/smp/audit_smp_prior_quality.py")


def test_compute_window_statistics_reports_distribution_and_smoothness():
    module = _load_audit_module()
    windows = np.zeros((2, 3, 6), dtype=np.float32)
    windows[:, :, 0] = np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32)
    windows[:, :, 3] = np.array([[0.0, 2.0, 4.0], [4.0, 6.0, 8.0]], dtype=np.float32)

    stats = module.compute_window_statistics(windows)

    assert stats["num_windows"] == 2
    assert stats["window_size"] == 3
    assert stats["feature_dim"] == 6
    assert stats["feature_mean_abs_mean"] > 0.0
    assert stats["temporal_diff_l2_mean"] > 0.0
    assert stats["temporal_acc_l2_mean"] == 0.0


def test_compare_generated_to_reference_detects_close_distribution():
    module = _load_audit_module()
    reference = np.zeros((4, 5, 8), dtype=np.float32)
    generated = reference + 0.01

    comparison = module.compare_window_statistics(
        module.compute_window_statistics(reference),
        module.compute_window_statistics(generated),
    )

    assert comparison["feature_mean_l2"] > 0.0
    assert comparison["feature_mean_l2"] < 0.1
    assert comparison["temporal_diff_l2_ratio"] == 1.0
    assert comparison["temporal_acc_l2_ratio"] == 1.0


def test_build_pass_fail_flags_prior_quality_ordering():
    module = _load_audit_module()

    flags = module.build_pass_fail_flags(
        group_summaries={
            "train": {"raw_noise_mse_mean": 0.3, "reward_mean": 0.37},
            "val": {"raw_noise_mse_mean": 0.33, "reward_mean": 0.35},
            "negative": {"raw_noise_mse_mean": 0.6, "reward_mean": 0.2},
        },
        generated_comparison={
            "feature_mean_l2": 0.2,
            "feature_std_l2": 0.3,
            "temporal_diff_l2_ratio": 1.2,
            "temporal_acc_l2_ratio": 1.4,
        },
        max_val_train_mse_ratio=1.5,
        max_generated_mean_l2=1.0,
        max_generated_std_l2=1.0,
        min_temporal_ratio=0.5,
        max_temporal_ratio=2.0,
    )

    assert flags["val_close_to_train"] is True
    assert flags["negative_worse_than_val"] is True
    assert flags["generated_mean_close"] is True
    assert flags["generated_std_close"] is True
    assert flags["generated_temporal_diff_reasonable"] is True
    assert flags["generated_temporal_acc_reasonable"] is True
    assert flags["overall"] is True
