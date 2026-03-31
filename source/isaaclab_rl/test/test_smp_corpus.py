# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_module(relative_parts: tuple[str, ...], module_name: str):
    # 在父目录链中回溯定位源码文件，避免依赖固定工作目录。
    for parent in Path(__file__).resolve().parents:
        module_path = parent.joinpath(*relative_parts)
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module
    raise FileNotFoundError(f"Could not find module: {'/'.join(relative_parts)}")


def _load_smp_corpus_module():
    return _load_module(("rsl_rl", "rsl_rl", "motion", "smp_corpus.py"), "isaaclab_smp_corpus_unit")


def _load_export_corpus_module():
    return _load_module(
        ("scripts", "imitation_learning", "smp", "export_g1_motion_corpus.py"),
        "isaaclab_smp_export_corpus_unit",
    )


def test_style_manifest_builds_stable_label_mapping_with_multi_shards_per_style(tmp_path):
    # 同一种风格可绑定多个 shard，且 style_id 需要按稳定排序生成。
    smp_corpus = _load_smp_corpus_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "walk_a", "path": "a.npz", "style": "walk"},
                    {"name": "walk_b", "path": "b.npz", "style": "walk"},
                    {"name": "dance_a", "path": "c.npz", "style": "dance"},
                ]
            }
        ),
        encoding="utf-8",
    )

    corpus = smp_corpus.SMPMotionCorpus.from_manifest(manifest_path)

    assert corpus.style_to_id == {"dance": 0, "walk": 1}
    assert [entry.name for entry in corpus.entries if entry.style_name == "walk"] == ["walk_a", "walk_b"]


def test_export_g1_motion_corpus_writes_manifest_and_style_vocab(tmp_path):
    # 语料导出脚本应把多个原始/预处理文件整理成统一语料目录。
    exporter = _load_export_corpus_module()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"

    for name in ("walk_a", "walk_b", "dance_a"):
        np.savez(
            input_dir / f"{name}.npz",
            fps=np.array([30], dtype=np.float32),
            frames=np.random.randn(12, 131).astype(np.float32),
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "walk_a", "path": str(input_dir / "walk_a.npz"), "style": "walk"},
                    {"name": "walk_b", "path": str(input_dir / "walk_b.npz"), "style": "walk"},
                    {"name": "dance_a", "path": str(input_dir / "dance_a.npz"), "style": "dance"},
                ]
            }
        ),
        encoding="utf-8",
    )

    exporter.export_g1_motion_corpus(manifest_path=manifest_path, output_dir=output_dir, window_size=10, stride=2)

    manifest_out = json.loads((output_dir / "corpus_manifest.json").read_text(encoding="utf-8"))
    style_vocab = json.loads((output_dir / "style_vocab.json").read_text(encoding="utf-8"))

    assert style_vocab == {"dance": 0, "walk": 1}
    assert {entry["name"] for entry in manifest_out["datasets"]} == {"walk_a", "walk_b", "dance_a"}
    assert all((output_dir / entry["path"]).is_file() for entry in manifest_out["datasets"])
    walk_entries = [entry for entry in manifest_out["datasets"] if entry["style"] == "walk"]
    assert {entry["style_id"] for entry in walk_entries} == {1}
