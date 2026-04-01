# SMP 多风格条件扩散与 GSI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 G1 SMP 扩散先验实现上，增加“多数据集多风格标签条件训练”、基于 classifier-free guidance 的风格条件奖励，以及利用冻结扩散模型生成动作初始状态分布的 GSI 流程；并支持同一种风格对应多个 `npz` 数据片段联合训练，同时在 prior 参数冻结后通过 cfg 传入单风格标签或基于身体掩码的多风格组合标签。

**Architecture:** 保留当前“离线训练扩散先验，在线 PPO 仅加载冻结 prior”的总体结构，不把扩散模型和策略网络端到端耦合。离线侧把原有单一 `frames` 数据集扩展为“多数据源 + style 标签 + manifest 元数据”的条件扩散训练链，其中 manifest 允许多个 `npz` shard 共享同一个 `style_name`，这些 shard 在训练时映射到同一个 `style_id`。在线侧把现有 unconditional `eps_theta(x_t, t)` 扩展为 `eps_theta(x_t, t, style)` 与 `eps_theta(x_t, t, null)` 两条前向。冻结后 cfg 只暴露两种明确模式：`single-style` 和 `body-mask composition`。`single-style` 直接走论文里的 CFG 风格特化；`body-mask composition` 则按论文 Figure 6 的思路，在 prior 的 `epsilon` 预测特征维上用身体部位掩码做组合：`f_comp = Σ M_part ⊙ f(x_i, c_part)`，例如上肢掩码对应 style `a`，下肢掩码对应 style `c`。这里的 mask 必须落到 `smp_motion_window` 的特征索引，而不是 PPO action space。GSI 采样复用同一套 style program，但在接入环境 reset 之前必须先验证当前 SMP 特征窗口是否足以恢复 reset 所需 simulator state。

**Tech Stack:** IsaacLab manager-based locomotion envs, nested `rsl_rl`, PyTorch, TensorBoard, NumPy, Hydra, Gymnasium

---

## Scope And Constraints

- 当前代码基线已经具备：
  - 单数据集 `frames -> sliding window -> unconditional diffusion prior`
  - 在线 SMP reward 与 TensorBoard 噪声误差日志
- 当前代码尚不具备：
  - 风格标签输入链路
  - classifier-free guidance
  - 反向扩散采样
  - 从 SMP 特征窗口恢复 simulator reset state 的解码链路
- 本计划按两个可独立验收的里程碑组织：
  - 里程碑 A：多风格条件扩散 + 风格条件 SMP reward
  - 里程碑 B：GSI 初始状态生成与环境 reset 集成
- GSI 是风险最高的部分，因为当前 SMP 特征并不是完整 simulator state，必须额外定义“哪些特征足够恢复 reset state、缺失量如何补全”。
- 下游风格控制的第一版语义定义为：
  - `mode=single_style` 时，整个 prior 使用一个 style label；这个 style 在离线训练阶段可以对应多个 `npz` shard；
  - `mode=body_mask` 时，通过 `body_part_style_names={upper_body:a, lower_body:c}` 这类 cfg 显式声明各身体部位对应的 style；
  - 身体部位掩码最终要映射到 `smp_motion_window` 的特征索引，不直接对 PPO action space 做掩码。
- 新增/修改的 Python 代码注释统一使用中文，重点解释风格解析、CFG 组合、GSI 回退与状态恢复等非显然逻辑。

## Paper Alignment Corrections

- 论文中的风格特化路径是“一个冻结的条件 diffusion 模型 + 下游训练时通过 style label / CFG 做 specialization”，因此 cfg 应该控制单风格 prior 或显式的组合程序，而不是在冻结前后再复制多份 style-specific checkpoint。
- 论文支持风格合成，但核心是对不同身体部位的 `epsilon` 预测做掩码组合。对本项目第一版，`a+c` 应解释为“身体掩码组合程序”，而不是风格池采样，也不是对多个条件输出做无结构平均。
- 论文中的 GSI 是“从 pretrained prior 采样初始状态分布”。由于当前 IsaacLab 实现训练的是 `smp_motion_window` 特征而不是完整 simulator state，落地时必须增加 reset-state codec 可逆性验证门；若验证失败，先扩展导出特征，再做 GSI reset 集成。
- 论文 Figure 6 给出的组合式是直接对不同 style-conditioned 预测做身体掩码混合：
  `f_comp = M_upper ⊙ f(x_i, c_a) + M_lower ⊙ f(x_i, c_c)`。
  因此本计划中的 `body_mask` 默认必须在 `guidance_scale=1.0` 时退化到完全相同的形式；若后续希望对组合 prior 继续加大/减弱风格强度，则允许把组合后的 `f_cond_comp` 再与 `f(x_i, ∅)` 做一层 generalized CFG，这属于对论文公式的向后兼容扩展，而不是替代其主体语义。

## G1 Feature Layout And `g1_upper_lower` Mask Template

当前 `pack_smp_frame_features()` 的单帧特征维度固定为 `131`，拼接顺序已经由现有测试锁定：

| 零基区间 | 维度 | 来源 | 模板归属 | 说明 |
| --- | ---: | --- | --- | --- |
| `[0:3)` | 3 | `base_lin_vel_b` | `shared_body` | 根线速度，属于全身共享状态 |
| `[3:6)` | 3 | `base_ang_vel_b` | `shared_body` | 根角速度，属于全身共享状态 |
| `[6:35)` | 29 | `joint_pos_rel` | 运行时按关节名拆分 | 不能直接写死“前 12 维腿、后 14 维手臂” |
| `[35:47)` | 12 | `ee_pos_b` | 按 `g1_ee_names` 拆分 | 双脚归 `lower_body`，双手归 `upper_body` |
| `[47:131)` | 84 | `key_body_rot6d` | 按 `g1_key_body_names` 拆分 | 骨盆/躯干共享，腿链归下肢，手臂链归上肢 |

`g1_upper_lower` 模板的第一版不要只做两组，而是内部固定生成三组互斥掩码：

- `shared_body`
- `lower_body`
- `upper_body`

原因是 `base_*`、骨盆、躯干和腰部关节同时影响上下肢，若硬塞到单一肢体会把掩码语义写死。为兼容你期望的用户配置形式，cfg 仍然可以只传：

```python
body_part_style_names = {"upper_body": "a", "lower_body": "c"}
```

但 runner 在解析 `mask_name=g1_upper_lower` 时需要额外补全 `shared_body`：

- 若 cfg 显式提供 `shared_body`，则直接使用该 style。
- 若 cfg 未提供 `shared_body`，第一版默认继承 `lower_body` 的 style，因为 locomotion 任务中的根速度、骨盆和腰部动态更强地受步态控制。
- 这个默认值必须写入日志与 checkpoint metadata，避免下游误以为模板只有上下肢两组。

`joint_pos_rel` 的 `29` 维需要按 G1 的关节名在运行时求索引，而不是把裸索引硬编码到模板里。当前 `unitree.py` 暴露的 `joint_sdk_names` 顺序可作为基线：

- `lower_body` 关节名：
  `left_hip_pitch_joint`, `left_hip_roll_joint`, `left_hip_yaw_joint`, `left_knee_joint`, `left_ankle_pitch_joint`, `left_ankle_roll_joint`, `right_hip_pitch_joint`, `right_hip_roll_joint`, `right_hip_yaw_joint`, `right_knee_joint`, `right_ankle_pitch_joint`, `right_ankle_roll_joint`
- `shared_body` 关节名：
  `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint`
- `upper_body` 关节名：
  `left_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `left_shoulder_yaw_joint`, `left_elbow_joint`, `left_wrist_roll_joint`, `left_wrist_pitch_joint`, `left_wrist_yaw_joint`, `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_shoulder_yaw_joint`, `right_elbow_joint`, `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint`

`ee_pos_b` 的模板映射按 `g1_ee_names` 固定：

- `[35:38)` `left_ankle_roll_link` -> `lower_body`
- `[38:41)` `right_ankle_roll_link` -> `lower_body`
- `[41:44)` `left_wrist_roll_link` -> `upper_body`
- `[44:47)` `right_wrist_roll_link` -> `upper_body`

`key_body_rot6d` 的模板映射按 `g1_key_body_names` 固定，每个 body 占 `6` 维：

- `[47:53)` `pelvis` -> `shared_body`
- `[53:59)` `torso_link` -> `shared_body`
- `[59:65)` `left_hip_pitch_link` -> `lower_body`
- `[65:71)` `right_hip_pitch_link` -> `lower_body`
- `[71:77)` `left_knee_link` -> `lower_body`
- `[77:83)` `right_knee_link` -> `lower_body`
- `[83:89)` `left_ankle_roll_link` -> `lower_body`
- `[89:95)` `right_ankle_roll_link` -> `lower_body`
- `[95:101)` `left_shoulder_pitch_link` -> `upper_body`
- `[101:107)` `right_shoulder_pitch_link` -> `upper_body`
- `[107:113)` `left_elbow_link` -> `upper_body`
- `[113:119)` `right_elbow_link` -> `upper_body`
- `[119:125)` `left_wrist_roll_link` -> `upper_body`
- `[125:131)` `right_wrist_roll_link` -> `upper_body`

模板构建与验证规则必须写进实现计划：

- mask 由关节名和 body 名推导，不直接硬编码原始索引常量。
- 产出的三个 mask 都是长度 `131` 的 `bool/float` 向量，并在窗口维上广播。
- 必须通过“互斥 + 完全覆盖”校验：`shared + lower + upper == 1`。
- 必须把 `joint_name_order`、`g1_ee_names`、`g1_key_body_names`、`feature_block_offsets`、`mask_template_name`、`mask_template_version` 一起存进 checkpoint metadata，确保冻结 prior 后下游 cfg 组合的特征语义不漂移。

## File Map

- Create `docs/smp-diffusion-code-map.md` - 补充多风格条件训练、CFG 奖励、GSI 采样的新代码链说明。
- Create `docs/smp-diffusion-end-to-end.md` - 更新端到端流程图，覆盖多数据集导出、条件 prior、GSI reset。
- Create `scripts/imitation_learning/smp/export_g1_motion_corpus.py` - 多数据集导出入口，读取 manifest 并写出统一训练语料。
- Create `scripts/imitation_learning/smp/export_g1_motion_dataset.py` - 下沉公共导出函数，支持单文件导出时附带 `style_name/style_id`。
- Create `scripts/imitation_learning/smp/train_motion_prior.py` - 增加 corpus/manifest、style 配置、CFG dropout、EMA/checkpoint 元数据参数。
- Create `rsl_rl/rsl_rl/motion/smp_corpus.py` - 定义多风格数据集 manifest 解析、同风格多 shard 聚合、style vocab、窗口索引。
- Modify `rsl_rl/rsl_rl/motion/smp_dataset.py` - 从仅返回 `window` 扩展为返回 `window + style_id + clip metadata`。
- Modify `rsl_rl/rsl_rl/motion/__init__.py` - 导出新数据集与 manifest 工具。
- Modify `rsl_rl/rsl_rl/diffusion/model.py` - 把 unconditional epsilon transformer 扩展为 style-conditioned 网络，并支持 null style。
- Create `rsl_rl/rsl_rl/diffusion/conditioning.py` - 封装 style embedding、conditioning dropout、CFG 组合函数。
- Create `rsl_rl/rsl_rl/diffusion/composition.py` - 封装身体部位特征掩码、style 组合程序解析与 `epsilon` 预测组合函数。
- Modify `rsl_rl/rsl_rl/diffusion/trainer.py` - 接入条件训练、style dropout、style-aware logging、checkpoint 元数据。
- Modify `rsl_rl/rsl_rl/diffusion/logging.py` - 增加 per-style loss、cond/uncond loss、CFG 诊断指标。
- Modify `rsl_rl/rsl_rl/diffusion/__init__.py` - 导出新增 conditioning 与 sampler 模块。
- Create `rsl_rl/rsl_rl/diffusion/sampler.py` - 实现 DDPM 反向采样、CFG 采样、窗口级生成接口。
- Create `rsl_rl/rsl_rl/diffusion/gsi.py` - 从采样窗口解码 reset 状态，校验姿态可用性，并在失败时回退。
- Modify `rsl_rl/rsl_rl/diffusion/smp_reward.py` - 增加基于 cond/uncond 预测的 CFG 奖励接口。
- Modify `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py` - 在线 reward 使用 style-conditioned prior；可选启用 GSI reset 统计。
- Modify `rsl_rl/rsl_rl/runners/__init__.py` - 保持 runner 导出一致。
- Create `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py` - 新增 `style_cfg`、`gsi_cfg` 等配置，避免把 SMP 专用字段直接塞进通用 `rl_cfg.py`。
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py` - 定义 G1 身体部位分组、身体掩码模板名、以及 GSI 所需的状态字段维度常量。
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py` - 在 G1 SMP 任务中接入默认 style 条件与 GSI 配置。
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py` - 视需要注册 style-specific task id 或保持单任务 + Hydra 覆盖。
- Create `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py` - 增加 reset state 编解码相关的公共特征工具。
- Create `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_reset.py` - 环境 reset 所需状态组装与应用工具。
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py` - 导出 `smp_features` 与 `smp_reset`。
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` - 暴露 GSI reset 开关与 reset 钩子。
- Create `source/isaaclab_rl/test/test_smp_corpus.py` - 多风格 manifest 与标签映射测试。
- Create `source/isaaclab_rl/test/test_smp_dataset.py` - 覆盖 style_id 返回值与多数据集滑窗。
- Create `source/isaaclab_rl/test/test_smp_diffusion_core.py` - 覆盖条件前向、null style、CFG 组合。
- Create `source/isaaclab_rl/test/test_smp_style_composition.py` - 覆盖身体掩码组合公式、掩码互斥性与特征覆盖率检查。
- Create `source/isaaclab_rl/test/test_smp_sampler.py` - 覆盖反向采样与窗口形状。
- Create `source/isaaclab_rl/test/test_smp_gsi.py` - 覆盖 reset 状态解码、合法性过滤与回退。
- Create `source/isaaclab_rl/test/test_smp_reward_logging.py` - 覆盖 per-style / cond-uncond / body-mask composition reward TensorBoard tag。

## Code Chain To Preserve

```text
多源原始 motion npz + 风格标签 manifest
  -> export_g1_motion_corpus.py
    -> export_g1_motion_dataset.py
      -> pack_smp_frame_features()
  -> corpus.npz / manifest.json
  -> SMPStyleMotionWindowDataset
  -> train_motion_prior.py
    -> SMPDiffusionTrainer
      -> MotionEpsilonTransformer(xt, t, style_id, drop_style)
      -> ExponentialMovingAverage
      -> SummaryWriter(style-aware tags)
  -> model_latest.pt
    -> 包含 style_vocab / cond_cfg / sampler_cfg
  -> SMPOnPolicyRunner
    -> _load_prior_model()
    -> _resolve_style_program_from_cfg()
    -> eps_uncond = prior(xt, t, null_style)
    -> eps_cond = prior(xt, t, style_id)
    -> eps_comp = compose_style_predictions_with_body_masks(part_to_eps, feature_masks)
    -> SMPReward.compute_from_cfg(...)
    -> TensorBoard: SMP/style/*, SMP/style_program/*, SMP/cfg/*
  -> sampler.py / gsi.py
    -> sample_window(style_program)
    -> decode_reset_state(window)
    -> env.reset_to_generated_state(...)
```

### Task 1: Add Multi-Style Corpus Format And Dataset Loader

**Files:**
- Create: `scripts/imitation_learning/smp/export_g1_motion_corpus.py`
- Create: `scripts/imitation_learning/smp/export_g1_motion_dataset.py`
- Create: `rsl_rl/rsl_rl/motion/smp_corpus.py`
- Modify: `rsl_rl/rsl_rl/motion/smp_dataset.py`
- Modify: `rsl_rl/rsl_rl/motion/__init__.py`
- Test: `source/isaaclab_rl/test/test_smp_corpus.py`
- Test: `source/isaaclab_rl/test/test_smp_dataset.py`

- [ ] **Step 1: 写一个失败的 manifest 解析测试**

```python
def test_style_manifest_builds_stable_label_mapping_with_multi_shards_per_style(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {"name": "walk_a", "path": "a.npz", "style": "walk"},
                    {"name": "walk_b", "path": "b.npz", "style": "walk"},
                    {"name": "dance_a", "path": "c.npz", "style": "dance"},
                ]
            }
        ),
        encoding="utf-8",
    )
    corpus = SMPMotionCorpus.from_manifest(manifest_path)
    assert corpus.style_to_id == {"dance": 0, "walk": 1}
    assert [entry.name for entry in corpus.entries if entry.style_name == "walk"] == ["walk_a", "walk_b"]
```

- [ ] **Step 2: 跑测试确认当前实现不支持 style manifest**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_corpus.py -q`
Expected: FAIL with missing `SMPMotionCorpus` or missing style mapping support.

- [ ] **Step 3: 定义统一语料格式**

```python
@dataclass
class SMPDatasetEntry:
    name: str
    path: Path
    style_name: str
    weight: float = 1.0
```

```python
class SMPMotionCorpus:
    def __init__(self, entries: list[SMPDatasetEntry]):
        self.entries = entries
        self.style_names = sorted({entry.style_name for entry in entries})
        self.style_to_id = {name: idx for idx, name in enumerate(self.style_names)}
        self.entries_by_style = {
            style_name: [entry for entry in entries if entry.style_name == style_name]
            for style_name in self.style_names
        }
```

- [ ] **Step 4: 修改导出脚本支持同风格多 shard manifest 输入**

Run: `python scripts/imitation_learning/smp/export_g1_motion_corpus.py --manifest /tmp/manifest.json --output /tmp/g1_styles`
Expected: 输出目录中生成：
- `corpus_manifest.json`
- `dataset_*.npz`
- `style_vocab.json`
并且 `walk_a.npz/walk_b.npz` 这类 shard 会共享同一个 `style_id`。

- [ ] **Step 5: 修改数据集类返回 `(window, style_id, meta)`**

```python
sample = dataset[index]
assert set(sample.keys()) == {"motion", "style_id", "style_name", "clip_id", "source_name"}
```

- [ ] **Step 6: 运行数据集测试**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_dataset.py source/isaaclab_rl/test/test_smp_corpus.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/imitation_learning/smp/export_g1_motion_corpus.py \
        scripts/imitation_learning/smp/export_g1_motion_dataset.py \
        rsl_rl/rsl_rl/motion/smp_corpus.py \
        rsl_rl/rsl_rl/motion/smp_dataset.py \
        rsl_rl/rsl_rl/motion/__init__.py \
        source/isaaclab_rl/test/test_smp_corpus.py \
        source/isaaclab_rl/test/test_smp_dataset.py
git commit -m "feat: add multi-style SMP corpus format"
```

### Task 2: Extend The Diffusion Model To Support Style Conditioning

**Files:**
- Modify: `rsl_rl/rsl_rl/diffusion/model.py`
- Create: `rsl_rl/rsl_rl/diffusion/conditioning.py`
- Create: `rsl_rl/rsl_rl/diffusion/composition.py`
- Modify: `rsl_rl/rsl_rl/diffusion/trainer.py`
- Modify: `rsl_rl/rsl_rl/diffusion/logging.py`
- Create: `scripts/imitation_learning/smp/train_motion_prior.py`
- Test: `source/isaaclab_rl/test/test_smp_diffusion_core.py`

- [ ] **Step 1: 写失败的条件前向测试**

```python
def test_motion_epsilon_transformer_accepts_style_ids():
    model = MotionEpsilonTransformer(
        feature_dim=131,
        window_size=10,
        num_diffusion_steps=50,
        num_styles=4,
    )
    xt = torch.randn(2, 10, 131)
    t = torch.tensor([22, 15], dtype=torch.long)
    style_id = torch.tensor([1, 3], dtype=torch.long)
    eps_hat = model(xt, t, style_id=style_id)
    assert eps_hat.shape == xt.shape
```

- [ ] **Step 2: 跑测试确认当前模型只支持 unconditional 前向**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_diffusion_core.py -q`
Expected: FAIL with unexpected keyword argument `style_id` or missing conditioning path.

- [ ] **Step 3: 在条件模块中实现 style embedding 与 null-style 约定**

```python
NULL_STYLE_ID = -1

def maybe_drop_style(style_id: torch.Tensor, drop_prob: float) -> torch.Tensor:
    mask = torch.rand_like(style_id.float()) < drop_prob
    dropped = style_id.clone()
    dropped[mask] = NULL_STYLE_ID
    return dropped
```

- [ ] **Step 4: 修改模型前向签名与 checkpoint 配置**

```python
def forward(self, xt, t, style_id=None):
    style_embed = self.style_conditioner(style_id)
    hidden = hidden + timestep_embed + style_embed.unsqueeze(1)
    return self.output_proj(self.encoder(hidden))
```

- [ ] **Step 5: 修改 trainer，训练时显式混入无条件样本**

Run: `python scripts/imitation_learning/smp/train_motion_prior.py --dataset /tmp/corpus_manifest.json --logdir /tmp/smp_style --style-drop-prob 0.1 --num-styles 4`
Expected: 训练正常启动，并在 TensorBoard 中出现：
- `SMPPretrain/loss_total`
- `SMPPretrain/loss_cond`
- `SMPPretrain/loss_uncond`
- `SMPPretrain/style/<style_name>/noise_mse`

- [ ] **Step 6: 保存完整 style 元数据到 checkpoint**

```python
checkpoint["style_cfg"] = {
    "style_names": dataset.style_names,
    "style_to_id": dataset.style_to_id,
    "null_style_id": NULL_STYLE_ID,
    "drop_prob": self.style_drop_prob,
}
checkpoint["feature_layout_cfg"] = {
    "feature_names": dataset.feature_names,
    "feature_block_offsets": dataset.feature_block_offsets,
    "joint_name_order": dataset.joint_name_order,
    "ee_name_order": dataset.ee_name_order,
    "key_body_name_order": dataset.key_body_name_order,
    "body_part_feature_indices": dataset.body_part_feature_indices,
    "mask_template_name": dataset.mask_template_name,
    "mask_template_version": dataset.mask_template_version,
}
```

- [ ] **Step 7: 运行条件扩散核心测试**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_diffusion_core.py -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add rsl_rl/rsl_rl/diffusion/model.py \
        rsl_rl/rsl_rl/diffusion/conditioning.py \
        rsl_rl/rsl_rl/diffusion/composition.py \
        rsl_rl/rsl_rl/diffusion/trainer.py \
        rsl_rl/rsl_rl/diffusion/logging.py \
        scripts/imitation_learning/smp/train_motion_prior.py \
        source/isaaclab_rl/test/test_smp_diffusion_core.py
git commit -m "feat: add style-conditioned SMP diffusion training"
```

### Task 3: Add CFG-Based Single-Style Reward And Body-Mask Style Composition At PPO Time

**Files:**
- Modify: `rsl_rl/rsl_rl/diffusion/smp_reward.py`
- Create: `rsl_rl/rsl_rl/diffusion/composition.py`
- Modify: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- Create: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`
- Create: `source/isaaclab_rl/test/test_smp_style_composition.py`
- Create: `source/isaaclab_rl/test/test_smp_reward_logging.py`

- [ ] **Step 1: 写失败的身体掩码组合奖励测试**

```python
def test_body_mask_style_composition_blends_feature_groups():
    eps_a = torch.tensor([[[1.0, 1.0, 1.0, 1.0]]])
    eps_c = torch.tensor([[[9.0, 9.0, 9.0, 9.0]]])
    upper_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    lower_mask = torch.tensor([[[0.0, 0.0, 1.0, 1.0]]])
    eps_comp = compose_style_predictions_with_body_masks(
        {"upper_body": eps_a, "lower_body": eps_c},
        {"upper_body": upper_mask, "lower_body": lower_mask},
    )
    assert torch.allclose(eps_comp, torch.tensor([[[1.0, 1.0, 9.0, 9.0]]]))
```

```python
def test_g1_upper_lower_mask_template_is_disjoint_and_exhaustive():
    masks = build_g1_body_part_feature_masks(
        mask_name="g1_upper_lower",
        joint_name_order=joint_name_order,
        ee_name_order=g1_ee_names,
        key_body_name_order=g1_key_body_names,
        feature_block_offsets=feature_block_offsets,
    )
    stacked = torch.stack(
        [masks["shared_body"], masks["lower_body"], masks["upper_body"]],
        dim=0,
    )
    assert torch.all(stacked.sum(dim=0) == 1)
```

- [ ] **Step 2: 跑测试确认当前 reward 只接受单路 `eps_hat`**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_style_composition.py source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: FAIL with missing body-mask composition helper or missing style-conditioned reward path.

- [ ] **Step 3: 在 runner 中显式计算三组量**

```python
if self.style_cfg.mode == "single_style":
    eps_uncond_t = self.smp_prior(xt, t, style_id=null_style)
    eps_cond_t = self.smp_prior(xt, t, style_id=target_style_id)
    eps_prior_t = apply_classifier_free_guidance(eps_uncond_t, eps_cond_t, self.style_cfg.guidance_scale)
else:
    eps_uncond_t = self.smp_prior(xt, t, style_id=null_style)
    eps_part_t = {
        part_name: self.smp_prior(xt, t, style_id=style_id)
        for part_name, style_id in self.style_program.part_style_ids.items()
    }
    eps_cond_comp_t = compose_style_predictions_with_body_masks(eps_part_t, self.style_program.feature_masks)
    eps_prior_t = apply_classifier_free_guidance(
        eps_uncond_t,
        eps_cond_comp_t,
        self.style_cfg.guidance_scale,
    )
```

Expected behavior:
- 当 `mode=body_mask` 且 `guidance_scale=1.0` 时，公式严格退化为论文 Figure 6 的 `M_upper ⊙ f(x_i, c_a) + M_lower ⊙ f(x_i, c_c)`；
- 当 `guidance_scale != 1.0` 时，等价于在组合后的条件预测 `eps_cond_comp_t` 外再套一层 generalized CFG，这属于对论文公式的可选扩展。

- [ ] **Step 4: 在配置中加入单风格与身体掩码组合两种 style program**

```python
@configclass
class SMPStyleCfg:
    mode: str = "single_style"  # single_style | body_mask
    target_style_name: str = "walk"
    target_style_id: int | None = None
    guidance_scale: float = 1.0
    mask_name: str = "g1_upper_lower"
    shared_style_name: str | None = None
    body_part_style_names: dict[str, str] = {"upper_body": "a", "lower_body": "c"}
```

Expected behavior:
- `mode=single_style` 时，整段 prior 使用一个 style label；
- `mode=body_mask` 时，runner 解析 `body_part_style_names`，若模板存在 `shared_body` 且 cfg 未显式提供，则默认令 `shared_body -> lower_body`；
- `mode=body_mask` 时，runner 对每个部位单独前向，先得到 `eps_cond_comp`，再与 `eps_uncond` 组合，保证 `guidance_scale=1.0` 时与论文公式完全一致；
- `mask_name` 对应的身体掩码必须能映射到 `smp_motion_window` 的特征索引，并通过互斥性与覆盖率校验。

- [ ] **Step 5: 为 TensorBoard 增加风格条件诊断指标**

Run: `tensorboard --logdir /home/lucas/logs/rsl_rl/g1_smp`
Expected: 可见新增 tags：
- `SMP/style/mode`
- `SMP/style/target_id`
- `SMP/style_program/shared_body_style_id`
- `SMP/style_program/upper_body_style_id`
- `SMP/style_program/lower_body_style_id`
- `SMP/style_mask/coverage`
- `SMP/cfg/noise_mse`
- `SMP/cfg/cond_uncond_gap`
- `SMP/style/<style_name>/noise_mse`

- [ ] **Step 6: 运行奖励日志测试**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_style_composition.py source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add rsl_rl/rsl_rl/diffusion/smp_reward.py \
        rsl_rl/rsl_rl/diffusion/composition.py \
        rsl_rl/rsl_rl/runners/smp_on_policy_runner.py \
        source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py \
        source/isaaclab_rl/test/test_smp_style_composition.py \
        source/isaaclab_rl/test/test_smp_reward_logging.py
git commit -m "feat: add style-guided SMP reward"
```

### Task 4: Validate Reset-State Representation, Then Add Reverse Diffusion Sampling And Window-Level GSI

**Files:**
- Create: `rsl_rl/rsl_rl/diffusion/sampler.py`
- Create: `rsl_rl/rsl_rl/diffusion/gsi.py`
- Modify: `rsl_rl/rsl_rl/diffusion/__init__.py`
- Create: `source/isaaclab_rl/test/test_smp_sampler.py`
- Create: `source/isaaclab_rl/test/test_smp_gsi.py`

Gate:
- 只有在“采样窗口 -> reset state -> 重新编码回 SMP 特征”的误差达到可接受范围后，才继续 Task 5 的环境 reset 集成。
- 若 gate 失败，先回到 Task 1/Task 2 扩展导出特征，不直接硬接 GSI。

- [ ] **Step 1: 写失败的反向采样测试**

```python
def test_ddpm_sampler_returns_motion_window_shape():
    sampler = SMPDiffusionSampler(dummy_model, num_diffusion_steps=50, feature_dim=131, window_size=10)
    sample = sampler.sample(batch_size=4, style_id=torch.tensor([1, 1, 2, 3]))
    assert sample.shape == (4, 10, 131)
```

- [ ] **Step 2: 跑测试确认当前代码没有采样器**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_sampler.py -q`
Expected: FAIL with missing `SMPDiffusionSampler`.

- [ ] **Step 3: 实现论文对齐的 DDPM 反向采样接口**

```python
class SMPDiffusionSampler:
    def sample(self, batch_size, style_id, guidance_scale=1.0):
        xt = torch.randn(batch_size, self.window_size, self.feature_dim, device=self.device)
        for timestep in reversed(range(self.num_diffusion_steps)):
            xt = self.p_sample(xt, timestep, style_id=style_id, guidance_scale=guidance_scale)
        return xt
```

- [ ] **Step 4: 把窗口采样结果解码成 reset 所需状态**

```python
@dataclass
class SMPResetState:
    root_pos_w: torch.Tensor
    root_quat_w: torch.Tensor
    root_lin_vel_w: torch.Tensor
    root_ang_vel_w: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
```

- [ ] **Step 5: 定义最小可行 GSI 解码策略**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_gsi.py -q`
Expected: PASS，并验证：
- 从采样窗口取最后一帧恢复 `joint_pos`
- 根姿态由 `key_body_quat_b + root frame` 恢复，若不够稳定则回退到默认站立朝向
- 缺失的 `joint_vel` / `root_pos_w` 使用环境默认值或零速度补全
- 若上述恢复误差超过阈值，则显式报告“当前 `smp_motion_window` 不足以支持 GSI”，并回退到补充特征导出任务

- [ ] **Step 6: Commit**

```bash
git add rsl_rl/rsl_rl/diffusion/sampler.py \
        rsl_rl/rsl_rl/diffusion/gsi.py \
        rsl_rl/rsl_rl/diffusion/__init__.py \
        source/isaaclab_rl/test/test_smp_sampler.py \
        source/isaaclab_rl/test/test_smp_gsi.py
git commit -m "feat: add SMP reverse sampler and GSI decoder"
```

### Task 5: Integrate GSI With IsaacLab Reset Flow

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_reset.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Modify: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`

- [ ] **Step 1: 写失败的 reset 集成测试或 smoke test**

```python
def test_generated_reset_state_can_be_applied_to_env(env):
    reset_state = build_dummy_reset_state(env.num_envs)
    apply_smp_reset_state(env, env_ids=torch.arange(env.num_envs), state=reset_state)
```

- [ ] **Step 2: 跑 smoke test，确认当前 env reset 没有 GSI 分支**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_gsi.py -q`
Expected: FAIL with missing `apply_smp_reset_state` or reset hook.

- [ ] **Step 3: 在配置中定义 GSI 运行模式**

```python
@configclass
class SMPGSICfg:
    enabled: bool = False
    sample_on_reset: bool = True
    guidance_scale: float = 1.0
    fallback_to_default_reset: bool = True
    max_resample_attempts: int = 3
```

- [ ] **Step 4: 在 runner 中接管 reset 前状态生成**

```python
if self.gsi_cfg.enabled and len(reset_env_ids) > 0:
    sampled_state = self.gsi_sampler.sample_reset_state(
        len(reset_env_ids),
        style_program=self.style_program,
    )
    apply_smp_reset_state(self.env, reset_env_ids, sampled_state)
```

- [ ] **Step 5: 为 GSI 加入运行监控**

Run: `tensorboard --logdir /home/lucas/logs/rsl_rl/g1_smp`
Expected: 可见：
- `SMP/GSI/reset_accept_rate`
- `SMP/GSI/reset_resample_count`
- `SMP/GSI/style_program`
- `SMP/GSI/fallback_rate`

- [ ] **Step 6: 做最小真机链路 smoke test**

Run: `python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-SMP-Velocity-Flat-G1-v0 --agent rsl_rl_cfg_entry_point --num_envs 64 agent.smp_prior.checkpoint_path=/abs/model_latest.pt agent.gsi_cfg.enabled=true`
Expected: 训练能启动，首次 reset 不报 shape 或 physics state 赋值错误。

- [ ] **Step 7: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_reset.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py \
        rsl_rl/rsl_rl/runners/smp_on_policy_runner.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py
git commit -m "feat: integrate SMP GSI with env resets"
```

### Task 6: Update Docs, Commands, And Verification Coverage

**Files:**
- Create: `docs/smp-diffusion-code-map.md`
- Create: `docs/smp-diffusion-end-to-end.md`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py`
- Modify: `source/isaaclab_rl/test/test_smp_reward_logging.py`
- Modify: `source/isaaclab_rl/test/test_smp_style_composition.py`
- Modify: `source/isaaclab_rl/test/test_smp_diffusion_core.py`
- Modify: `source/isaaclab_rl/test/test_smp_dataset.py`

- [ ] **Step 1: 更新文档，明确三条运行命令**

Run:

```bash
python scripts/imitation_learning/smp/export_g1_motion_corpus.py --manifest /abs/manifest.json --output /abs/smp_corpus
python scripts/imitation_learning/smp/train_motion_prior.py --dataset /abs/smp_corpus/corpus_manifest.json --logdir /abs/logs/smp_prior/g1_style
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-SMP-Velocity-Flat-G1-v0 --agent rsl_rl_cfg_entry_point agent.smp_prior.checkpoint_path=/abs/model_latest.pt agent.smp_prior.style_cfg.mode=body_mask "agent.smp_prior.style_cfg.body_part_style_names={upper_body:a,lower_body:c}" agent.smp_prior.style_cfg.mask_name=g1_upper_lower
```

- [ ] **Step 2: 在任务注册层决定是否增加 style-specific task id**

Expected decision:
- 默认推荐保持单一 task id `Isaac-SMP-Velocity-Flat-G1-v0`
- 通过 Hydra 覆盖 `mode/target_style_name/body_part_style_names/mask_name` 切换单风格或身体掩码组合风格
- 只有在需要固定 benchmark 配置时，再注册 `Isaac-SMP-Velocity-Flat-G1-Walk-v0` 之类的别名

- [ ] **Step 3: 汇总完整验证矩阵**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_corpus.py source/isaaclab_rl/test/test_smp_dataset.py source/isaaclab_rl/test/test_smp_diffusion_core.py source/isaaclab_rl/test/test_smp_style_composition.py source/isaaclab_rl/test/test_smp_sampler.py source/isaaclab_rl/test/test_smp_gsi.py source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: PASS.

- [ ] **Step 4: 做端到端人工验收**

Expected checklist:
- 多个数据集能被导出到统一 corpus
- 同一种风格的多个 `npz` shard 会共享同一个 `style_id`
- 训练 checkpoint 含 style vocab
- TensorBoard 能看到 per-style 噪声误差
- 在线 reward 能切换单风格与上/下肢掩码组合风格
- `g1_upper_lower` 模板通过互斥性与完全覆盖校验
- 开启 GSI 后 reset 流程仍稳定

- [ ] **Step 5: Commit**

```bash
git add docs/smp-diffusion-code-map.md \
        docs/smp-diffusion-end-to-end.md \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py \
        source/isaaclab_rl/test/test_smp_reward_logging.py \
        source/isaaclab_rl/test/test_smp_style_composition.py \
        source/isaaclab_rl/test/test_smp_diffusion_core.py \
        source/isaaclab_rl/test/test_smp_dataset.py
git commit -m "docs: document multi-style SMP and GSI workflow"
```

## Recommended Implementation Order

1. 先完成 Task 1 和 Task 2，只做离线多风格条件扩散训练。
2. 然后完成 Task 3，把在线 reward 切到 CFG 条件 prior。
3. 只有前两段稳定后再做 Task 4 和 Task 5，因为 GSI 牵涉 simulator reset 和物理状态合法性。
4. Task 6 最后做，确保文档、命令和测试矩阵与最终实现一致。

## Key Technical Decisions

- **风格标签粒度**
  - 以“数据集 style 标签”为第一版，不在同一轨迹内部再切更细 style token。
- **同风格多片段语义**
  - 同一个 `style_name` 可以绑定多个 `npz` shard，并在离线训练时共享同一个 `style_id`。
  - shard 是数据组织粒度，不是新的风格粒度；不能因为文件分片而额外产生新的 style label。
- **条件训练方式**
  - 第一版使用 style embedding + conditioning dropout。
  - 不在第一轮就重写成复杂 AdaLN Transformer，先做兼容现有 `MotionEpsilonTransformer` 的最小修改。
- **风格程序语义**
  - `mode=single_style` 时，沿用论文里的单风格特化路径。
  - `mode=body_mask` 时，cfg 通过 `body_part_style_names` 显式声明每个身体部位对应的风格，例如 `upper_body:a, lower_body:c`。
  - `g1_upper_lower` 模板在实现上必须拆成 `shared_body/lower_body/upper_body` 三组，其中 `shared_body` 默认继承 `lower_body`，但允许 cfg 覆盖。
  - 组合发生在 prior 的 `epsilon` 预测特征维，不直接发生在 PPO action space。
- **CFG 使用位置**
  - 训练时通过 style dropout 学到 cond/uncond。
  - `mode=single_style` 时直接对 `eps_uncond` 与 `eps_cond` 做 CFG。
  - `mode=body_mask` 时先得到组合后的 `eps_cond_comp`，再与 `eps_uncond` 做 CFG；当 `guidance_scale=1.0` 时退化为论文 Figure 6 的原式。
- **GSI 最小解**
  - 先通过 reset-state codec 验证当前 SMP 特征能否近似恢复 reset 所需状态。
  - 先只恢复 reset 真正需要的状态字段。
  - 对无法从 SMP 特征稳定反解的量，使用默认站姿或零速度补全。
- **代码注释语言**
  - 新增/修改的 Python 代码统一使用中文注释，只解释非显然逻辑，不写噪声式注释。
- **任务暴露方式**
  - 优先保持一个 `Isaac-SMP-Velocity-Flat-G1-v0` 任务，通过 Hydra 指定 `mode/target_style_name/body_part_style_names/mask_name`。
  - 这样最少改动现有训练/播放命令，也更适合多风格 checkpoint 复用。

## Risks To Resolve During Implementation

- 从当前 SMP 特征能否稳定恢复根姿态和关节速度，需要先用离线回放做误差评估。
- 多风格数据集若帧数分布很不均衡，需要在 corpus 层支持采样权重，否则训练会偏向大数据集风格。
- 即使在同一种风格内部，不同 shard 的帧数也可能严重不均衡，需要支持 shard-level weight，否则同风格长片段会压制短片段。
- 身体部位掩码必须与 `smp_motion_window` 的特征布局严格对齐；若某些特征同时混合上下肢或根状态信息，就不能简单做硬掩码。
- `g1_upper_lower` 不能只返回上下肢两组；共享根状态、腰部和躯干若没有显式模板归属，组合 prior 会在这些维度上出现语义空洞。
- `upper_body:a + lower_body:c` 的组合必须落在 prior 特征维；若误做成 action-space mask，会偏离论文方法并破坏 prior 语义。
- GSI 采样出的状态可能不满足接触或高度约束，需要回退与重采样机制。
- `rsl_rl` 是嵌套仓库，真正的扩散模型、runner、reward 修改会出现在主项目根目录下的 `rsl_rl/`，不是当前 worktree 里的同名路径。

## Minimal Deliverables For Milestone A

- 多风格 manifest 导出脚本
- 条件扩散训练与 checkpoint style metadata / feature_layout metadata
- cfg 驱动的单风格与身体掩码 style program 解析
- `g1_upper_lower` 共享/上肢/下肢模板与元数据持久化
- CFG 单风格 prior 与 body-mask 组合 prior reward
- TensorBoard per-style / cond-uncond 指标

## Minimal Deliverables For Milestone B

- DDPM 反向采样器
- 与 body-mask composition 兼容的 style-program 采样接口
- reset-state codec 可逆性验证门
- 采样窗口到 reset state 的解码器
- 环境 reset 集成与 fallback
- GSI 运行监控指标
