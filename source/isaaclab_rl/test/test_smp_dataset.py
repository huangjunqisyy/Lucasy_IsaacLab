# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_smp_dataset_module():
    # 在当前测试文件的父目录链中查找源码模块，避免依赖固定工作目录。
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "rsl_rl" / "rsl_rl" / "motion" / "smp_dataset.py"
        if module_path.exists():
            # 以文件路径动态加载模块，便于在隔离测试环境下直接调用实现。
            spec = importlib.util.spec_from_file_location("isaaclab_smp_dataset_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find rsl_rl/rsl_rl/motion/smp_dataset.py")


def _load_export_g1_motion_dataset_module():
    # 动态定位并加载导出脚本模块，用于测试脚本逻辑而非命令行入口。
    for parent in Path(__file__).resolve().parents:
        module_path = parent / "scripts" / "imitation_learning" / "smp" / "export_g1_motion_dataset.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("isaaclab_smp_export_unit", module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError("Could not find scripts/imitation_learning/smp/export_g1_motion_dataset.py")


def test_smp_dataset_builds_sliding_windows(tmp_path):
    # 构造最小可复现数据，验证滑窗数量与单个窗口形状是否符合预期。
    smp_dataset = _load_smp_dataset_module()
    path = tmp_path / "toy_motion.npz"
    np.savez(
        path,
        fps=np.array([30]),
        frames=np.random.randn(12, 131).astype(np.float32),
    )

    dataset = smp_dataset.SMPMotionWindowDataset(path, window_size=10, stride=1)

    assert len(dataset) == 3
    sample = dataset[0]
    assert set(sample.keys()) == {"motion", "style_id", "style_name", "clip_id", "source_name"}
    assert sample["motion"].shape == (10, 131)
    assert sample["style_id"] is None
    assert sample["style_name"] is None


def test_smp_dataset_reads_multi_style_manifest_and_returns_metadata(tmp_path):
    # manifest 模式下应返回风格标签、来源名和 clip 元数据。
    smp_dataset = _load_smp_dataset_module()
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    np.savez(data_dir / "walk_a.npz", fps=np.array([30]), frames=np.random.randn(12, 131).astype(np.float32))
    np.savez(data_dir / "walk_b.npz", fps=np.array([30]), frames=np.random.randn(13, 131).astype(np.float32))
    manifest_path = data_dir / "corpus_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "walk_a", "path": "walk_a.npz", "style": "walk", "style_id": 0},
                    {"name": "walk_b", "path": "walk_b.npz", "style": "walk", "style_id": 0},
                ]
            }
        ),
        encoding="utf-8",
    )

    dataset = smp_dataset.SMPMotionWindowDataset(manifest_path, window_size=10, stride=2)

    assert len(dataset) == 4
    first = dataset[0]
    last = dataset[-1]
    assert set(first.keys()) == {"motion", "style_id", "style_name", "clip_id", "source_name"}
    assert first["motion"].shape == (10, 131)
    assert first["style_id"] == 0
    assert first["style_name"] == "walk"
    assert first["source_name"] == "walk_a"
    assert last["source_name"] == "walk_b"


def test_export_g1_motion_dataset_writes_frames_and_metadata(tmp_path):
    # 验证导出脚本会写出 frames 以及训练所需的关键元数据字段。
    exporter = _load_export_g1_motion_dataset_module()
    input_path = tmp_path / "raw_motion.npz"
    output_path = tmp_path / "smp_frames.npz"
    body_quat_w = np.zeros((12, 30, 4), dtype=np.float32)
    body_quat_w[..., 0] = 1.0

    np.savez(
        input_path,
        fps=np.array([30], dtype=np.int64),
        joint_pos=np.zeros((12, 29), dtype=np.float32),
        joint_vel=np.zeros((12, 29), dtype=np.float32),
        body_pos_w=np.zeros((12, 30, 3), dtype=np.float32),
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((12, 30, 3), dtype=np.float32),
        body_ang_vel_w=np.zeros((12, 30, 3), dtype=np.float32),
    )

    exporter.export_g1_motion_dataset(
        input_path=input_path,
        output_path=output_path,
        window_size=10,
        stride=2,
        style_name="walk",
        style_id=3,
        source_name="walk_a",
    )

    with np.load(output_path) as data:
        assert data["frames"].shape == (12, 131)
        assert int(data["window_size"][0]) == 10
        assert int(data["stride"][0]) == 2
        assert int(data["feature_dim"][0]) == 131
        assert str(data["style_name"].reshape(-1)[0]) == "walk"
        assert int(data["style_id"].reshape(-1)[0]) == 3
        assert str(data["source_name"].reshape(-1)[0]) == "walk_a"


def test_export_g1_motion_dataset_requires_window_size_argument():
    # 覆盖 CLI 参数约束：缺少必填 window-size 时应触发解析失败。
    exporter = _load_export_g1_motion_dataset_module()
    parser = exporter._build_argparser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--input", "in.npz", "--output", "out.npz"])
