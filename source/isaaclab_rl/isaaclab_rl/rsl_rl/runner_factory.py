# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module


def _import_string(path: str):
    module_name, attr_name = path.split(":", 1) if ":" in path else (None, None)
    if not module_name or not attr_name:
        raise ValueError(f"runner_type must be in 'module:attr' format: {path}")
    module = import_module(module_name)
    return getattr(module, attr_name)


def resolve_runner_class(agent_cfg):
    runner_type = getattr(agent_cfg, "runner_type", None)
    if runner_type:
        return _import_string(runner_type) if isinstance(runner_type, str) else runner_type
    class_name = getattr(agent_cfg, "class_name", None)
    if class_name == "OnPolicyRunner":
        from rsl_rl.runners import OnPolicyRunner

        return OnPolicyRunner
    if class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        return DistillationRunner
    raise ValueError(f"Unsupported runner class: {class_name}")
