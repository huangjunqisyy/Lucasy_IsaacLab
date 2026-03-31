# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
import types
from pathlib import Path

import torch

from rsl_rl.diffusion.gsi import SMPResetState


def _load_smp_reset_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "isaaclab_tasks"
        / "isaaclab_tasks"
        / "manager_based"
        / "locomotion"
        / "velocity"
        / "mdp"
        / "smp_reset.py"
    )
    spec = importlib.util.spec_from_file_location("isaaclab_smp_reset_unit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeRobot:
    def __init__(self):
        self.data = types.SimpleNamespace(
            default_root_state=torch.tensor(
                [
                    [0.0, 0.0, 0.74, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [1.0, 0.0, 0.80, 1.0, 0.0, 0.0, 0.0, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
                    [2.0, 0.0, 0.90, 1.0, 0.0, 0.0, 0.0, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
                ],
                dtype=torch.float32,
            ),
            default_joint_pos=torch.arange(12, dtype=torch.float32).view(3, 4),
            default_joint_vel=torch.full((3, 4), 0.25, dtype=torch.float32),
        )
        self.root_pose_call = None
        self.root_velocity_call = None
        self.joint_state_call = None
        self.joint_position_target_call = None
        self.joint_velocity_target_call = None

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids=None):
        self.root_pose_call = (root_pose.clone(), env_ids.clone())

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids=None):
        self.root_velocity_call = (root_velocity.clone(), env_ids.clone())

    def write_joint_state_to_sim(self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids=None):
        self.joint_state_call = (joint_pos.clone(), joint_vel.clone(), env_ids.clone())

    def set_joint_position_target(self, joint_pos: torch.Tensor, env_ids=None):
        self.joint_position_target_call = (joint_pos.clone(), env_ids.clone())

    def set_joint_velocity_target(self, joint_vel: torch.Tensor, env_ids=None):
        self.joint_velocity_target_call = (joint_vel.clone(), env_ids.clone())


class _FakeEnv:
    def __init__(self, robot: _FakeRobot):
        self.scene = {"robot": robot}
        self.unwrapped = self


def test_build_smp_reset_reference_reads_robot_defaults():
    smp_reset = _load_smp_reset_module()
    env = _FakeEnv(_FakeRobot())

    reference = smp_reset.build_smp_reset_reference(env, env_ids=torch.tensor([0, 2]))

    assert torch.allclose(reference.root_pos_w, torch.tensor([[0.0, 0.0, 0.74], [2.0, 0.0, 0.90]]))
    assert torch.allclose(reference.root_quat_w, torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(reference.joint_pos, torch.tensor([[0.0, 1.0, 2.0, 3.0], [8.0, 9.0, 10.0, 11.0]]))
    assert torch.allclose(reference.joint_vel, torch.full((2, 4), 0.25))


def test_apply_smp_reset_state_writes_pose_velocity_and_joint_targets():
    smp_reset = _load_smp_reset_module()
    robot = _FakeRobot()
    env = _FakeEnv(robot)
    env_ids = torch.tensor([1, 2], dtype=torch.long)

    state = SMPResetState(
        root_pos_w=torch.tensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]], dtype=torch.float32),
        root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        root_lin_vel_w=torch.tensor([[0.4, 0.5, 0.6], [1.4, 1.5, 1.6]], dtype=torch.float32),
        root_ang_vel_w=torch.tensor([[0.7, 0.8, 0.9], [1.7, 1.8, 1.9]], dtype=torch.float32),
        joint_pos=torch.tensor([[0.0, 0.1, 0.2, 0.3], [1.0, 1.1, 1.2, 1.3]], dtype=torch.float32),
        joint_vel=torch.tensor([[0.9, 0.8, 0.7, 0.6], [1.9, 1.8, 1.7, 1.6]], dtype=torch.float32),
    )

    smp_reset.apply_smp_reset_state(env, env_ids=env_ids, state=state)

    assert torch.allclose(
        robot.root_pose_call[0],
        torch.tensor(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [1.1, 1.2, 1.3, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.allclose(
        robot.root_velocity_call[0],
        torch.tensor(
            [
                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                [1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            ]
        ),
    )
    assert torch.equal(robot.root_pose_call[1], env_ids)
    assert torch.equal(robot.root_velocity_call[1], env_ids)
    assert torch.equal(robot.joint_state_call[2], env_ids)
    assert torch.equal(robot.joint_position_target_call[1], env_ids)
    assert torch.equal(robot.joint_velocity_target_call[1], env_ids)
    assert torch.allclose(robot.joint_state_call[0], state.joint_pos)
    assert torch.allclose(robot.joint_state_call[1], state.joint_vel)
