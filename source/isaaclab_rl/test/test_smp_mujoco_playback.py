import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_playback_module():
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "scripts" / "imitation_learning" / "smp" / "smp_mujoco_playback_utils.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_mujoco_playback_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find scripts/imitation_learning/smp/smp_mujoco_playback_utils.py")


class _DummyExtractor:
    def decode_joint_positions(self, windows: torch.Tensor) -> torch.Tensor:
        batch, steps, _ = windows.shape
        return torch.zeros((batch, steps, 29), dtype=windows.dtype, device=windows.device)


def test_load_csv_motion_to_trajectory_preserves_pose_columns(tmp_path):
    module = _load_playback_module()

    raw = np.zeros((4, 36), dtype=np.float32)
    raw[:, 0] = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    raw[:, 2] = 0.75
    raw[:, 6] = 1.0
    raw[:, 7:36] = 0.25
    csv_path = tmp_path / "toy.csv"
    np.savetxt(csv_path, raw, delimiter=",")

    trajectory = module.load_csv_trajectory(csv_path=csv_path, fps=30.0)

    assert trajectory.source_mode == "csv_raw"
    assert trajectory.fps == 30.0
    assert trajectory.root_pos.shape == (4, 3)
    assert trajectory.root_quat_xyzw.shape == (4, 4)
    assert trajectory.joint_pos.shape == (4, 29)
    np.testing.assert_allclose(trajectory.root_pos[:, 0], raw[:, 0])
    np.testing.assert_allclose(trajectory.root_quat_xyzw[:, 3], 1.0)
    np.testing.assert_allclose(trajectory.joint_pos, 0.25)


def test_decode_windows_features_to_trajectory_returns_expected_shapes(tmp_path):
    module = _load_playback_module()

    windows = np.zeros((2, 10, 192), dtype=np.float32)
    windows[:, :, 0] = 0.2
    windows[:, :, 5] = 0.1
    npz_path = tmp_path / "toy_windows.npz"
    np.savez(npz_path, windows=windows, feature_fps=np.array([50.0], dtype=np.float32))

    trajectory = module.load_npz_trajectory(
        npz_path=npz_path,
        mode="window",
        sample_index=0,
        window_index=1,
        extractor=_DummyExtractor(),
        default_height=0.74,
        fps_override=None,
    )

    assert trajectory.source_mode == "windows[1]"
    assert trajectory.root_pos.shape == (10, 3)
    assert trajectory.root_quat_xyzw.shape == (10, 4)
    assert trajectory.joint_pos.shape == (10, 29)
    np.testing.assert_allclose(trajectory.root_pos[:, 2], 0.74)


def test_integrate_root_from_features_accumulates_planar_motion():
    module = _load_playback_module()

    features = np.zeros((3, 192), dtype=np.float32)
    features[:, 0] = 1.0
    root_pos, root_quat = module.integrate_root_from_features(features, fps=2.0, default_height=0.8)

    np.testing.assert_allclose(root_pos[:, 2], 0.8)
    np.testing.assert_allclose(root_pos[:, 0], np.array([0.0, 0.5, 1.0], dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(root_quat[:, 3], 1.0, atol=1e-6)


def test_joint_qpos_indices_accepts_joint_names_that_already_have_suffix(monkeypatch):
    module = _load_playback_module()

    class _FakeMujoco:
        class mjtObj:
            mjOBJ_JOINT = object()

        @staticmethod
        def mj_name2id(model, _obj_type, joint_name):
            return model.name_to_id.get(joint_name, -1)

    class _FakeModel:
        def __init__(self):
            joint_names = module.DATASET_JOINT_NAMES
            self.name_to_id = {name: idx for idx, name in enumerate(joint_names)}
            self.jnt_qposadr = np.arange(len(joint_names), dtype=np.int32)

    monkeypatch.setitem(sys.modules, "mujoco", _FakeMujoco)

    joint_indices = module._joint_qpos_indices(_FakeModel())

    np.testing.assert_array_equal(joint_indices, np.arange(len(module.DATASET_JOINT_NAMES), dtype=np.int32))


def test_load_feature_window_batch_from_input_returns_contiguous_windows(tmp_path):
    module = _load_playback_module()

    windows = np.arange(4 * 10 * 192, dtype=np.float32).reshape(4, 10, 192)
    npz_path = tmp_path / "toy_windows.npz"
    np.savez(npz_path, windows=windows, feature_fps=np.array([50.0], dtype=np.float32))

    selected, fps, source_mode = module.load_feature_window_batch_from_input(
        input_path=npz_path,
        mode="window",
        csv_fps=30.0,
        sample_index=0,
        window_index=1,
        num_windows=2,
        window_size=10,
        stride=1,
        urdf_path=None,
    )

    assert selected.shape == (2, 10, 192)
    np.testing.assert_array_equal(selected, windows[1:3])
    assert fps == 50.0
    assert source_mode == "windows[1:3]"
