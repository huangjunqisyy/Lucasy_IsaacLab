# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


def test_resolve_runner_type_prefers_explicit_runner_type():
    from isaaclab_rl.rsl_rl.runner_factory import resolve_runner_class

    class DummyCfg:
        runner_type = "rsl_rl.runners:SMPOnPolicyRunner"
        class_name = "OnPolicyRunner"

    runner_cls = resolve_runner_class(DummyCfg())
    assert runner_cls.__name__ == "SMPOnPolicyRunner"
