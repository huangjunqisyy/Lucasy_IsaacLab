# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

_DEFAULT_PACKAGE_NAMES = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_mimic",
    "isaaclab_rl",
    "isaaclab_tasks",
)
_CORE_PACKAGE_NAMES = ("isaaclab", "isaaclab_rl", "isaaclab_tasks")


def find_repo_root(script_path: str | Path, package_names: Iterable[str] = _CORE_PACKAGE_NAMES) -> Path:
    """从脚本位置向上查找 Isaac Lab 仓库根目录。"""
    required_packages = tuple(package_names)
    current = Path(script_path).resolve()
    current = current if current.is_dir() else current.parent
    for parent in (current, *current.parents):
        source_dir = parent / "source"
        if all((source_dir / package_name).exists() for package_name in required_packages):
            return parent
    raise FileNotFoundError(f"Could not locate Isaac Lab repo root for packages: {required_packages}")


def prepend_local_source_paths(
    script_path: str | Path,
    sys_path: list[str] | None = None,
    package_names: Iterable[str] = _DEFAULT_PACKAGE_NAMES,
) -> list[str]:
    """把当前 worktree 的 source 包路径前置到 sys.path，优先于主工作区安装路径。"""
    resolved_sys_path = sys.path if sys_path is None else sys_path
    package_names = tuple(package_names)
    repo_marker_packages = tuple(package_name for package_name in _CORE_PACKAGE_NAMES if package_name in package_names)
    if not repo_marker_packages:
        repo_marker_packages = package_names
    repo_root = find_repo_root(script_path, package_names=repo_marker_packages)
    local_paths = [
        str(repo_root / "source" / package_name)
        for package_name in package_names
        if (repo_root / "source" / package_name).exists()
    ]

    # 去掉已有重复项后整体前置，确保 import 时优先命中当前 worktree。
    remaining_paths = [entry for entry in resolved_sys_path if entry not in local_paths]
    resolved_sys_path[:] = local_paths + remaining_paths
    return local_paths


def find_nested_rsl_rl_repo_root(script_path: str | Path) -> Path:
    """定位当前脚本可见的内嵌 rsl_rl 仓库，worktree 缺失时回退到父工作区。"""
    repo_root = find_repo_root(script_path, package_names=_CORE_PACKAGE_NAMES)
    for candidate_root in (repo_root, *repo_root.parents):
        candidate = candidate_root / "rsl_rl"
        if (candidate / "rsl_rl" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate nested rsl_rl repository")
