# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest



def _load_csv_export_module():
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "scripts" / "imitation_learning" / "smp" / "convert_g1_csv_to_smp_dataset.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_g1_csv_export_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find scripts/imitation_learning/smp/convert_g1_csv_to_smp_dataset.py")



def test_load_g1_csv_motion_parses_root_quat_and_joint_columns(tmp_path):
    exporter = _load_csv_export_module()

    raw = np.arange(72, dtype=np.float32).reshape(2, 36)
    csv_path = tmp_path / "toy.csv"
    np.savetxt(csv_path, raw, delimiter=",")

    motion = exporter.load_g1_csv_motion(csv_path, fps=30.0)

    assert motion.root_pos.shape == (2, 3)
    assert motion.root_quat_xyzw.shape == (2, 4)
    assert motion.joint_pos.shape == (2, 29)
    np.testing.assert_allclose(motion.root_pos, raw[:, 0:3])
    np.testing.assert_allclose(motion.root_quat_xyzw, raw[:, 3:7])
    np.testing.assert_allclose(motion.joint_pos, raw[:, 7:36])



def test_build_smp_frames_from_motion_arrays_returns_192d(tmp_path):
    exporter = _load_csv_export_module()

    num_frames = 12
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    root_pos[:, 2] = 0.8
    root_quat_xyzw = np.zeros((num_frames, 4), dtype=np.float32)
    root_quat_xyzw[:, 3] = 1.0
    joint_pos = np.zeros((num_frames, 29), dtype=np.float32)
    ee_pos_w = np.zeros((num_frames, 4, 3), dtype=np.float32)

    frames = exporter.build_smp_frames_from_motion_arrays(
        root_pos=root_pos,
        root_quat_xyzw=root_quat_xyzw,
        joint_pos=joint_pos,
        ee_pos_w=ee_pos_w,
        fps=30.0,
    )
    output_path = tmp_path / "frames_dataset.npz"
    exporter.save_smp_frames_dataset(
        output_path=output_path,
        frames=frames,
        fps=30.0,
        window_size=10,
        stride=2,
        source_name="toy_clip",
    )

    assert frames.shape == (num_frames, 192)
    with np.load(output_path) as data:
        assert data["frames"].shape == (num_frames, 192)
        assert int(data["feature_dim"][0]) == 192
        assert int(data["window_size"][0]) == 10
        assert int(data["stride"][0]) == 2
        assert str(data["source_name"][0]) == "toy_clip"


def test_convert_single_csv_to_smp_dataset_with_urdf_backend(tmp_path):
    exporter = _load_csv_export_module()

    try:
        urdf_path = exporter._resolve_g1_urdf_path()
    except FileNotFoundError:
        pytest.skip("No local G1 URDF available for pure-URDF FK test")

    num_frames = 8
    raw = np.zeros((num_frames, 36), dtype=np.float32)
    raw[:, 2] = 0.8
    raw[:, 6] = 1.0
    csv_path = tmp_path / "toy_motion.csv"
    np.savetxt(csv_path, raw, delimiter=",")

    output_path = tmp_path / "toy_motion_smp.npz"
    summary = exporter.convert_single_csv_to_smp_dataset(
        csv_path=csv_path,
        output_path=output_path,
        source_fps=30.0,
        target_fps=None,
        fk_backend="urdf",
        urdf_path=urdf_path,
        verify=False,
    )

    assert summary.feature_dim == 192
    with np.load(output_path) as data:
        assert data["frames"].shape == (num_frames, 192)
        assert str(data["fk_backend"][0]) == "urdf"
