# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_runner_factory_module():
    module_path = Path(__file__).resolve().parents[1] / "isaaclab_rl" / "rsl_rl" / "runner_factory.py"
    spec = importlib.util.spec_from_file_location("isaaclab_rl_runner_factory_unit", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_runner_type_prefers_explicit_runner_type(monkeypatch):
    runner_factory = _load_runner_factory_module()

    fake_module = types.ModuleType("fake_runner_module")

    class FakeRunner:
        pass

    fake_module.FakeRunner = FakeRunner
    monkeypatch.setitem(sys.modules, "fake_runner_module", fake_module)

    class DummyCfg:
        runner_type = "fake_runner_module:FakeRunner"
        class_name = "OnPolicyRunner"

    runner_cls = runner_factory.resolve_runner_class(DummyCfg())
    assert runner_cls is FakeRunner


def test_resolve_runner_class_falls_back_to_class_name(monkeypatch):
    runner_factory = _load_runner_factory_module()

    fake_runners = types.ModuleType("rsl_rl.runners")

    class OnPolicyRunner:
        pass

    class DistillationRunner:
        pass

    fake_runners.OnPolicyRunner = OnPolicyRunner
    fake_runners.DistillationRunner = DistillationRunner
    monkeypatch.setitem(sys.modules, "rsl_rl.runners", fake_runners)

    class DummyCfg:
        class_name = "OnPolicyRunner"

    runner_cls = runner_factory.resolve_runner_class(DummyCfg())
    assert runner_cls is OnPolicyRunner


def test_resolve_runner_class_raises_for_unsupported_class():
    runner_factory = _load_runner_factory_module()

    class DummyCfg:
        class_name = "UnknownRunner"

    with pytest.raises(ValueError, match="Unsupported runner class: UnknownRunner"):
        runner_factory.resolve_runner_class(DummyCfg())
