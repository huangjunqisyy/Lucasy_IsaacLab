# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from importlib import import_module


class _LazyImportedRunner:
    def __init__(self, module_name: str, attr_name: str):
        self._module_name = module_name
        self._attr_name = attr_name
        self.__name__ = attr_name

    def __call__(self, *args, **kwargs):
        module = import_module(self._module_name)
        runner_class = getattr(module, self._attr_name)
        return runner_class(*args, **kwargs)


def _import_string(path: str):
    module_name, attr_name = path.split(":")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError:
        return _LazyImportedRunner(module_name, attr_name)
    return getattr(module, attr_name)


def resolve_runner_class(agent_cfg):
    runner_type = getattr(agent_cfg, "runner_type", None)
    if runner_type:
        return _import_string(runner_type) if isinstance(runner_type, str) else runner_type
    if agent_cfg.class_name == "OnPolicyRunner":
        from rsl_rl.runners import OnPolicyRunner

        return OnPolicyRunner
    if agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        return DistillationRunner
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
