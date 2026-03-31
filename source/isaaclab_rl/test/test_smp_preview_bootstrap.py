# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_bootstrap_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "imitation_learning"
        / "smp"
        / "path_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_smp_preview_bootstrap_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepend_local_source_paths_prefers_worktree_packages(tmp_path):
    bootstrap = _load_bootstrap_module()
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts" / "imitation_learning" / "smp" / "preview_gsi_resets.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# test script\n", encoding="utf-8")

    for package_name in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
        (repo_root / "source" / package_name).mkdir(parents=True)

    main_workspace_path = "/home/lucas/isaac-sim/IsaacLab/source"
    sys_path = [main_workspace_path, "/tmp/site-packages"]

    inserted_paths = bootstrap.prepend_local_source_paths(script_path=script_path, sys_path=sys_path)

    assert inserted_paths == [
        str(repo_root / "source" / "isaaclab"),
        str(repo_root / "source" / "isaaclab_rl"),
        str(repo_root / "source" / "isaaclab_tasks"),
    ]
    assert sys_path[:3] == inserted_paths
    assert sys_path[3:] == [main_workspace_path, "/tmp/site-packages"]


def test_prepend_local_source_paths_does_not_duplicate_existing_entries(tmp_path):
    bootstrap = _load_bootstrap_module()
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts" / "imitation_learning" / "smp" / "preview_gsi_resets.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# test script\n", encoding="utf-8")

    local_rl_path = repo_root / "source" / "isaaclab_rl"
    local_rl_path.mkdir(parents=True)
    sys_path = [str(local_rl_path), "/tmp/site-packages"]

    inserted_paths = bootstrap.prepend_local_source_paths(
        script_path=script_path,
        sys_path=sys_path,
        package_names=("isaaclab_rl",),
    )

    assert inserted_paths == [str(local_rl_path)]
    assert sys_path == [str(local_rl_path), "/tmp/site-packages"]


def test_find_nested_rsl_rl_repo_root_falls_back_to_parent_workspace(tmp_path):
    bootstrap = _load_bootstrap_module()
    workspace_root = tmp_path / "IsaacLab"
    worktree_root = workspace_root / ".worktrees" / "g1-smp-diffusion"
    script_path = worktree_root / "scripts" / "imitation_learning" / "smp" / "preview_gsi_resets.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# test script\n", encoding="utf-8")

    for package_name in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
        (worktree_root / "source" / package_name).mkdir(parents=True)

    nested_rsl_rl = workspace_root / "rsl_rl" / "rsl_rl"
    nested_rsl_rl.mkdir(parents=True)
    (nested_rsl_rl / "__init__.py").write_text("# nested repo\n", encoding="utf-8")

    resolved = bootstrap.find_nested_rsl_rl_repo_root(script_path)

    assert resolved == workspace_root / "rsl_rl"


def test_find_nested_rsl_rl_repo_root_raises_when_missing(tmp_path):
    bootstrap = _load_bootstrap_module()
    repo_root = tmp_path / "repo"
    script_path = repo_root / "scripts" / "imitation_learning" / "smp" / "preview_gsi_resets.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# test script\n", encoding="utf-8")

    for package_name in ("isaaclab", "isaaclab_rl", "isaaclab_tasks"):
        (repo_root / "source" / package_name).mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="nested rsl_rl repository"):
        bootstrap.find_nested_rsl_rl_repo_root(script_path)
