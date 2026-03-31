# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _find_rsl_rl_repo_root() -> Path:
    # 从当前脚本位置向上查找，定位内嵌的 rsl_rl 仓库根目录。
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "rsl_rl" / "rsl_rl" / "__init__.py"
        if candidate.exists():
            return parent / "rsl_rl"
    raise FileNotFoundError("Could not locate nested rsl_rl repository")


_RSL_RL_REPO_ROOT = _find_rsl_rl_repo_root()
# 将内嵌仓库加入 import 路径，确保可以直接导入 rsl_rl.diffusion。
if str(_RSL_RL_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_RSL_RL_REPO_ROOT))

from rsl_rl.diffusion import SMPDiffusionTrainer  # noqa: E402


def _build_argparser() -> argparse.ArgumentParser:
    # 统一管理命令行参数，便于脚本和批处理任务复用。
    parser = argparse.ArgumentParser(description="训练 G1 的 SMP motion diffusion prior。")
    parser.add_argument("--dataset", required=True, help="SMP 窗口数据集 npz 路径。")
    parser.add_argument("--logdir", required=True, help="TensorBoard 与 checkpoint 输出目录。")
    parser.add_argument("--batch-size", type=int, default=32, help="每次迭代的 batch size。")
    parser.add_argument("--max-iters", type=int, default=1000, help="离线预训练的最大迭代数。")
    parser.add_argument("--window-size", type=int, default=10, help="窗口长度。")
    parser.add_argument("--stride", type=int, default=1, help="训练数据的滑窗步长。")
    parser.add_argument("--num-diffusion-steps", type=int, default=50, help="扩散步数。")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Transformer 隐层维度。")
    parser.add_argument("--num-layers", type=int, default=2, help="Transformer 编码层数。")
    parser.add_argument("--num-heads", type=int, default=8, help="Transformer 注意力头数。")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4, help="优化器学习率。")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA 衰减系数。")
    parser.add_argument("--num-styles", type=int, default=None, help="条件模型可用的风格总数；默认从数据集推断。")
    parser.add_argument("--style-drop-prob", type=float, default=0.0, help="classifier-free guidance 的风格 dropout 概率。")
    parser.add_argument(
        "--timesteps-k",
        type=int,
        nargs="+",
        default=[22, 15, 8],
        help="训练与日志使用的固定扩散时间步集合。",
    )
    parser.add_argument("--device", default=None, help="显式指定训练设备，例如 cpu 或 cuda:0。")
    return parser


def main():
    # 解析参数并构建训练器，所有超参数均由命令行显式传入。
    args = _build_argparser().parse_args()
    trainer = SMPDiffusionTrainer(
        dataset_path=args.dataset,
        log_dir=args.logdir,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        window_size=args.window_size,
        stride=args.stride,
        num_diffusion_steps=args.num_diffusion_steps,
        timesteps_k=args.timesteps_k,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        ema_decay=args.ema_decay,
        num_styles=args.num_styles,
        style_drop_prob=args.style_drop_prob,
        device=args.device,
    )
    # 启动离线训练并输出关键结果，方便快速确认训练状态。
    result = trainer.train()
    print(f"训练完成，checkpoint: {result['checkpoint_path']}")
    print(f"最终 loss: {result['final_loss']:.6f}")


if __name__ == "__main__":
    main()
