# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(module_name: str, module_path: Path):
    # 通过文件路径加载模块，避免依赖包安装顺序。
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_workspace_root() -> Path:
    # 兼容从主仓库或隔离 worktree 运行脚本，统一定位到带有嵌套 rsl_rl 的根目录。
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "rsl_rl" / "rsl_rl" / "__init__.py"
        if candidate.exists():
            return parent
    raise FileNotFoundError("Could not locate workspace root with nested rsl_rl repository")


_WORKSPACE_ROOT = _find_workspace_root()
if str(_WORKSPACE_ROOT / "rsl_rl") not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT / "rsl_rl"))

from rsl_rl.motion import SMPMotionCorpus  # noqa: E402

_EXPORT_DATASET_MODULE = _load_module(
    "isaaclab_smp_export_dataset",
    Path(__file__).resolve().with_name("export_g1_motion_dataset.py"),
)
export_g1_motion_dataset = _EXPORT_DATASET_MODULE.export_g1_motion_dataset


def export_g1_motion_corpus(
    manifest_path: str | Path,
    output_dir: str | Path,
    window_size: int,
    stride: int = 1,
) -> Path:
    """将 manifest 中列出的多个数据片段导出为统一 SMP 语料目录。"""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = SMPMotionCorpus.from_manifest(manifest_path)
    exported_entries = []
    for entry in corpus.entries:
        output_name = f"dataset_{entry.name}.npz"
        export_g1_motion_dataset(
            input_path=entry.path,
            output_path=output_dir / output_name,
            window_size=window_size,
            stride=stride,
            style_name=entry.style_name,
            style_id=entry.style_id,
            source_name=entry.name,
        )
        exported_entries.append(
            {
                "name": entry.name,
                "path": output_name,
                "style": entry.style_name,
                "style_id": entry.style_id,
                "weight": entry.weight,
            }
        )

    (output_dir / "style_vocab.json").write_text(
        json.dumps(corpus.style_to_id, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    corpus_manifest = {
        "datasets": exported_entries,
        "style_vocab": corpus.style_to_id,
        "window_size": window_size,
        "stride": stride,
    }
    (output_dir / "corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_dir


def _build_argparser() -> argparse.ArgumentParser:
    # CLI 仅负责参数解析，导出逻辑集中到 export_g1_motion_corpus。
    parser = argparse.ArgumentParser(description="导出多风格 G1 SMP 语料目录。")
    parser.add_argument("--manifest", required=True, help="输入的多风格 manifest JSON 路径。")
    parser.add_argument("--output", required=True, help="统一语料输出目录。")
    parser.add_argument("--window-size", required=True, type=int, help="写入 metadata 的窗口大小。")
    parser.add_argument("--stride", type=int, default=1, help="写入 metadata 的滑窗步长。")
    return parser


def main():
    args = _build_argparser().parse_args()
    export_g1_motion_corpus(
        manifest_path=args.manifest,
        output_dir=args.output,
        window_size=args.window_size,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
