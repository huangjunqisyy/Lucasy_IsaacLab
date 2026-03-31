# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import torch


def _load_module(relative_parts: tuple[str, ...], module_name: str):
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


def test_body_mask_style_composition_blends_feature_groups():
    composition = _load_module(("rsl_rl", "rsl_rl", "diffusion", "composition.py"), "isaaclab_smp_composition_unit")
    eps_a = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    eps_c = torch.tensor([[[9.0, 9.0, 9.0, 9.0]]])
    upper_mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    lower_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])

    eps_comp = composition.compose_style_predictions_with_body_masks(
        {"upper_body": eps_a, "lower_body": eps_c},
        {"upper_body": upper_mask, "lower_body": lower_mask},
    )

    assert torch.allclose(eps_comp, torch.tensor([[[1.0, 1.0, 9.0, 9.0]]]))


def test_g1_upper_lower_mask_template_is_disjoint_and_exhaustive():
    composition = _load_module(("rsl_rl", "rsl_rl", "diffusion", "composition.py"), "isaaclab_smp_composition_unit")
    g1_config = _load_module(
        (
            ".worktrees",
            "g1-smp-diffusion",
            "source",
            "isaaclab_tasks",
            "isaaclab_tasks",
            "manager_based",
            "locomotion",
            "velocity",
            "config",
            "g1",
            "agents",
            "config.py",
        ),
        "isaaclab_g1_cfg_unit",
    )

    masks = composition.build_g1_body_part_feature_masks(
        mask_name=g1_config.g1_smp_mask_template_name,
        joint_name_order=g1_config.g1_smp_joint_names,
        ee_name_order=g1_config.g1_ee_names,
        key_body_name_order=g1_config.g1_key_body_names,
        feature_block_offsets=g1_config.g1_smp_feature_block_offsets,
    )
    stacked = torch.stack(
        [masks["shared_body"], masks["lower_body"], masks["upper_body"]],
        dim=0,
    )

    assert torch.all(stacked.sum(dim=0) == 1)
