# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
from pathlib import Path


def _load_bootstrap_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "imitation_learning"
        / "smp"
        / "path_bootstrap.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_train_bootstrap_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_bootstrap_helper_supports_train_script_path_layout(tmp_path):
    bootstrap = _load_bootstrap_module()
    workspace_root = tmp_path / "IsaacLab"
    worktree_root = workspace_root / ".worktrees" / "g1-smp-diffusion"
    script_path = worktree_root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# train script\n", encoding="utf-8")

    for package_name in ("isaaclab", "isaaclab_rl", "isaaclab_tasks", "isaaclab_assets"):
        (worktree_root / "source" / package_name).mkdir(parents=True)

    nested_rsl_rl = workspace_root / "rsl_rl" / "rsl_rl"
    nested_rsl_rl.mkdir(parents=True)
    (nested_rsl_rl / "__init__.py").write_text("# nested repo\n", encoding="utf-8")

    sys_path = ["/home/lucas/isaac-sim/IsaacLab/source"]
    inserted = bootstrap.prepend_local_source_paths(script_path=script_path, sys_path=sys_path)
    resolved_rsl_rl_root = bootstrap.find_nested_rsl_rl_repo_root(script_path)

    assert str(worktree_root / "source" / "isaaclab_rl") in inserted
    assert sys_path[0] == str(worktree_root / "source" / "isaaclab")
    assert resolved_rsl_rl_root == workspace_root / "rsl_rl"
