# G1 SMP Diffusion Prior Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-aligned SMP-style motion diffusion prior to IsaacLab's G1 manager-based locomotion stack, and expose TensorBoard metrics that show the gap between predicted noise and the true injected noise during both prior pretraining and online PPO training.

**Architecture:** Keep the diffusion model offline-trainable and frozen at policy-training time, matching the SMP paper's separation between a task-agnostic motion prior and the downstream controller. Reuse the existing G1 manager-based task and RSL-RL stack, add a dedicated `smp` observation window for motion features, and implement the SDS reward in a custom runner that logs scalar and histogram gap metrics to TensorBoard. Start with the paper's core path only: unconditional epsilon-prediction, fixed timestep ensemble `K={22,15,8}`, exponential reward shaping, and no CFG/GSI until the first end-to-end loop is stable.

**Tech Stack:** IsaacLab manager-based locomotion envs, RSL-RL PPO, PyTorch, TensorBoard, Hydra, Gymnasium, NumPy

---

## File Map

- Create `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py` - SMP runner config dataclasses (`SMPPriorCfg`, `SMPRunnerCfg`) without polluting `rl_cfg.py`.
- Create `source/isaaclab_rl/isaaclab_rl/rsl_rl/runner_factory.py` - single place that resolves `runner_type` or falls back to `class_name`.
- Modify `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py` - export SMP configs and runner factory helpers.
- Modify `scripts/reinforcement_learning/rsl_rl/train.py` - instantiate custom runner types instead of hardcoding only `OnPolicyRunner` and `DistillationRunner`.
- Create `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py` - shared pure-tensor feature packing utilities (`quat -> rot6d`, frame packing, window stacking).
- Create `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py` - online `smp_motion_window` observation term built from simulator state.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py` - export new observation utilities.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` - add an `smp` observation group for the runner.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py` - centralize feature constants such as key bodies, end-effectors, window size, and diffusion step defaults.
- Create `rsl_rl/rsl_rl/motion/smp_dataset.py` - offline window dataset loader and batch sampler for prior pretraining.
- Modify `rsl_rl/rsl_rl/motion/__init__.py` - export SMP dataset utilities.
- Create `scripts/imitation_learning/smp/export_g1_motion_dataset.py` - convert existing G1 motion `.npz` files into windowed SMP training data using the exact same feature packer as the online env.
- Create `rsl_rl/rsl_rl/diffusion/__init__.py` - diffusion package exports.
- Create `rsl_rl/rsl_rl/diffusion/scheduler.py` - beta schedule, `alpha_bar`, forward noising, timestep helpers.
- Create `rsl_rl/rsl_rl/diffusion/model.py` - compact transformer epsilon-predictor with timestep conditioning.
- Create `rsl_rl/rsl_rl/diffusion/ema.py` - EMA wrapper for prior pretraining.
- Create `rsl_rl/rsl_rl/diffusion/logging.py` - TensorBoard helper for scalar and histogram logging of `eps`, `eps_hat`, and `eps_gap`.
- Create `rsl_rl/rsl_rl/diffusion/trainer.py` - offline pretraining loop for the motion prior.
- Create `rsl_rl/rsl_rl/diffusion/smp_reward.py` - fixed-timestep SDS ensemble, adaptive per-timestep normalization, and reward shaping.
- Create `scripts/imitation_learning/smp/train_motion_prior.py` - CLI for prior pretraining and checkpoint export.
- Create `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py` - PPO runner that queries the frozen diffusion prior and logs noise-gap metrics.
- Modify `rsl_rl/rsl_rl/runners/__init__.py` - export `SMPOnPolicyRunner`.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py` - add `G1SMPRunnerCfg`.
- Modify `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py` - register `Isaac-SMP-Velocity-Flat-G1-v0`.
- Create `source/isaaclab_rl/test/test_rsl_rl_runner_factory.py` - pure unit tests for runner dispatch.
- Create `source/isaaclab_rl/test/test_smp_feature_utils.py` - pure tensor tests for feature packing.
- Create `source/isaaclab_rl/test/test_smp_dataset.py` - offline window slicing tests.
- Create `source/isaaclab_rl/test/test_smp_diffusion_core.py` - forward noising and model-shape tests.
- Create `source/isaaclab_rl/test/test_smp_reward_logging.py` - verify TensorBoard tags for noise-gap logging.
- Modify `source/isaaclab_tasks/test/test_hydra.py` - Hydra coverage for the new G1 SMP task.

### Task 1: Add SMP Runner Config And Runner Dispatch

**Files:**
- Create: `source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py`
- Create: `source/isaaclab_rl/isaaclab_rl/rsl_rl/runner_factory.py`
- Modify: `source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py`
- Modify: `scripts/reinforcement_learning/rsl_rl/train.py`
- Test: `source/isaaclab_rl/test/test_rsl_rl_runner_factory.py`

- [ ] **Step 1: Write the failing runner-dispatch test**

```python
def test_resolve_runner_type_prefers_explicit_runner_type():
    from isaaclab_rl.rsl_rl.runner_factory import resolve_runner_class

    class DummyCfg:
        runner_type = "rsl_rl.runners:SMPOnPolicyRunner"
        class_name = "OnPolicyRunner"

    runner_cls = resolve_runner_class(DummyCfg())
    assert runner_cls.__name__ == "SMPOnPolicyRunner"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest source/isaaclab_rl/test/test_rsl_rl_runner_factory.py -q`
Expected: FAIL with `ModuleNotFoundError` for `isaaclab_rl.rsl_rl.runner_factory` or missing `resolve_runner_class`.

- [ ] **Step 3: Implement minimal runner factory and SMP config surface**

```python
# source/isaaclab_rl/isaaclab_rl/rsl_rl/runner_factory.py
from importlib import import_module


def _import_string(path: str):
    module_name, attr_name = path.split(":")
    module = import_module(module_name)
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
```

```python
# source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py
from dataclasses import MISSING
from isaaclab.utils import configclass
from .rl_cfg import RslRlOnPolicyRunnerCfg


@configclass
class SMPPriorCfg:
    checkpoint_path: str = MISSING
    window_size: int = 10
    feature_dim: int = MISSING
    num_diffusion_steps: int = 50
    timesteps_k: list[int] = MISSING
    reward_scale: float = 1.0
    adaptive_norm_decay: float = 0.99
    log_histograms_every: int = 20


@configclass
class SMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    runner_type: str = "rsl_rl.runners:SMPOnPolicyRunner"
    smp_prior: SMPPriorCfg = MISSING
    smp_reward_coef: float = 1.0
    task_reward_coef: float = 1.0
```

- [ ] **Step 4: Wire `train.py` to use the new runner resolver**

Run: `python -m pytest source/isaaclab_rl/test/test_rsl_rl_runner_factory.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_rl/isaaclab_rl/rsl_rl/smp_cfg.py \
        source/isaaclab_rl/isaaclab_rl/rsl_rl/runner_factory.py \
        source/isaaclab_rl/isaaclab_rl/rsl_rl/__init__.py \
        scripts/reinforcement_learning/rsl_rl/train.py \
        source/isaaclab_rl/test/test_rsl_rl_runner_factory.py
git commit -m "feat: add SMP runner config and dispatch plumbing"
```

### Task 2: Add Shared SMP Feature Packing And Online Observation Window

**Files:**
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py`
- Create: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py`
- Test: `source/isaaclab_rl/test/test_smp_feature_utils.py`

- [ ] **Step 1: Write failing pure-tensor feature tests**

```python
def test_quat_to_rot6d_returns_six_values_per_body():
    quat = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32)
    rot6d = quat_to_rot6d(quat)
    assert rot6d.shape == (1, 1, 6)


def test_pack_smp_frame_features_has_expected_dim():
    features = pack_smp_frame_features(
        base_lin_vel_b=torch.zeros(2, 3),
        base_ang_vel_b=torch.zeros(2, 3),
        joint_pos_rel=torch.zeros(2, 29),
        ee_pos_b=torch.zeros(2, 4, 3),
        key_body_quat_b=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]] * 14).view(1, 14, 4).repeat(2, 1, 1),
    )
    assert features.shape == (2, 131)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_feature_utils.py -q`
Expected: FAIL with missing `quat_to_rot6d` and `pack_smp_frame_features`.

- [ ] **Step 3: Implement a single shared feature packer**

```python
def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    mat = matrix_from_quat(quat_wxyz.reshape(-1, 4)).reshape(*quat_wxyz.shape[:-1], 3, 3)
    return mat[..., :2].reshape(*quat_wxyz.shape[:-1], 6)


def pack_smp_frame_features(base_lin_vel_b, base_ang_vel_b, joint_pos_rel, ee_pos_b, key_body_quat_b):
    key_body_rot6d = quat_to_rot6d(key_body_quat_b).reshape(key_body_quat_b.shape[0], -1)
    ee_pos_b = ee_pos_b.reshape(ee_pos_b.shape[0], -1)
    return torch.cat([base_lin_vel_b, base_ang_vel_b, joint_pos_rel, ee_pos_b, key_body_rot6d], dim=-1)
```

- [ ] **Step 4: Expose an online `smp_motion_window` observation group**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_feature_utils.py -q`
Expected: PASS.

Run: `python -m pytest source/isaaclab_rl/test/test_rsl_rl_wrapper.py -q`
Expected: Existing wrapper tests still PASS after adding the `smp` observation group.

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/smp_features.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/my_observations.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/__init__.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/config.py \
        source/isaaclab_rl/test/test_smp_feature_utils.py
git commit -m "feat: add SMP feature packing and online observation window"
```

### Task 3: Add Offline Dataset Export And Windowed SMP Dataset Loader

**Files:**
- Create: `scripts/imitation_learning/smp/export_g1_motion_dataset.py`
- Create: `rsl_rl/rsl_rl/motion/smp_dataset.py`
- Modify: `rsl_rl/rsl_rl/motion/__init__.py`
- Test: `source/isaaclab_rl/test/test_smp_dataset.py`

- [ ] **Step 1: Write the failing dataset-windowing test**

```python
def test_smp_dataset_builds_sliding_windows(tmp_path):
    path = tmp_path / "toy_motion.npz"
    np.savez(
        path,
        fps=np.array([30]),
        frames=np.random.randn(12, 131).astype(np.float32),
    )
    dataset = SMPMotionWindowDataset(path, window_size=10, stride=1)
    assert len(dataset) == 3
    assert dataset[0].shape == (10, 131)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_dataset.py -q`
Expected: FAIL with missing `SMPMotionWindowDataset`.

- [ ] **Step 3: Implement the exporter and loader using the same feature packer as Task 2**

```python
class SMPMotionWindowDataset(torch.utils.data.Dataset):
    def __init__(self, path, window_size=10, stride=1):
        data = np.load(path)
        self.frames = torch.tensor(data["frames"], dtype=torch.float32)
        self.window_size = window_size
        self.start_ids = list(range(0, self.frames.shape[0] - window_size + 1, stride))

    def __len__(self):
        return len(self.start_ids)

    def __getitem__(self, index):
        start = self.start_ids[index]
        return self.frames[start : start + self.window_size]
```

- [ ] **Step 4: Validate export against the real G1 motion file**

Run: `python scripts/imitation_learning/smp/export_g1_motion_dataset.py --input source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/motion.npz --output /tmp/g1_smp_windows.npz --window-size 10`
Expected: `/tmp/g1_smp_windows.npz` created with `frames` and metadata fields.

Run: `python -m pytest source/isaaclab_rl/test/test_smp_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/imitation_learning/smp/export_g1_motion_dataset.py \
        rsl_rl/rsl_rl/motion/smp_dataset.py \
        rsl_rl/rsl_rl/motion/__init__.py \
        source/isaaclab_rl/test/test_smp_dataset.py
git commit -m "feat: add SMP motion dataset export and window loader"
```

### Task 4: Add Diffusion Core And Offline Prior Pretraining With TensorBoard

**Files:**
- Create: `rsl_rl/rsl_rl/diffusion/__init__.py`
- Create: `rsl_rl/rsl_rl/diffusion/scheduler.py`
- Create: `rsl_rl/rsl_rl/diffusion/model.py`
- Create: `rsl_rl/rsl_rl/diffusion/ema.py`
- Create: `rsl_rl/rsl_rl/diffusion/logging.py`
- Create: `rsl_rl/rsl_rl/diffusion/trainer.py`
- Create: `scripts/imitation_learning/smp/train_motion_prior.py`
- Test: `source/isaaclab_rl/test/test_smp_diffusion_core.py`
- Test: `source/isaaclab_rl/test/test_smp_reward_logging.py`

- [ ] **Step 1: Write the failing forward-noising and TensorBoard tag tests**

```python
def test_q_sample_matches_closed_form():
    scheduler = DiffusionScheduler(num_steps=50, beta_start=1e-4, beta_end=2e-2)
    x0 = torch.randn(4, 10, 131)
    eps = torch.randn_like(x0)
    t = torch.tensor([22, 15, 8, 22], dtype=torch.long)
    xt = scheduler.q_sample(x0, t, eps)
    alpha_bar = scheduler.alpha_bar[t].view(-1, 1, 1)
    expected = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * eps
    assert torch.allclose(xt, expected)
```

```python
def test_log_smp_pretrain_metrics_writes_noise_tags(tmp_path):
    writer = SummaryWriter(log_dir=tmp_path)
    log_smp_pretrain_metrics(
        writer,
        global_step=1,
        loss=0.5,
        per_timestep_mse={22: 0.6, 15: 0.4, 8: 0.3},
        eps=torch.zeros(8, 10, 131),
        eps_hat=torch.ones(8, 10, 131),
    )
    writer.flush()
    ea = EventAccumulator(str(tmp_path))
    ea.Reload()
    assert "SMPPretrain/noise_mse" in ea.Tags()["scalars"]
    assert "SMPPretrain/eps_gap" in ea.Tags()["histograms"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_diffusion_core.py source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: FAIL with missing diffusion classes and logging helpers.

- [ ] **Step 3: Implement the smallest paper-aligned prior**

```python
class DiffusionScheduler:
    def __init__(self, num_steps=50, beta_start=1e-4, beta_end=2e-2):
        self.beta = torch.linspace(beta_start, beta_end, num_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, eps):
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1).to(x0.device)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * eps
```

```python
class MotionEpsilonTransformer(nn.Module):
    def __init__(self, feature_dim, window_size, hidden_dim=256, num_layers=2):
        super().__init__()
        # timestep embedding + token projection + 2-layer transformer + epsilon head
```

- [ ] **Step 4: Run a tiny pretraining smoke job and inspect TensorBoard**

Run: `python scripts/imitation_learning/smp/train_motion_prior.py --dataset /tmp/g1_smp_windows.npz --logdir /tmp/smp_pretrain --batch-size 32 --max-iters 20 --num-diffusion-steps 50 --window-size 10`
Expected: TensorBoard event files plus a checkpoint under `/tmp/smp_pretrain`.

Run: `tensorboard --inspect --logdir /tmp/smp_pretrain`
Expected: tags include `SMPPretrain/noise_mse`, `SMPPretrain/t22/noise_mse`, `SMPPretrain/t15/noise_mse`, `SMPPretrain/t8/noise_mse`, and histogram tags for `eps`, `eps_hat`, and `eps_gap`.

- [ ] **Step 5: Commit**

```bash
git add rsl_rl/rsl_rl/diffusion/__init__.py \
        rsl_rl/rsl_rl/diffusion/scheduler.py \
        rsl_rl/rsl_rl/diffusion/model.py \
        rsl_rl/rsl_rl/diffusion/ema.py \
        rsl_rl/rsl_rl/diffusion/logging.py \
        rsl_rl/rsl_rl/diffusion/trainer.py \
        scripts/imitation_learning/smp/train_motion_prior.py \
        source/isaaclab_rl/test/test_smp_diffusion_core.py \
        source/isaaclab_rl/test/test_smp_reward_logging.py
git commit -m "feat: add SMP diffusion prior pretraining stack"
```

### Task 5: Add Frozen-Prior SDS Reward And Online SMP Runner

**Files:**
- Create: `rsl_rl/rsl_rl/diffusion/smp_reward.py`
- Create: `rsl_rl/rsl_rl/runners/smp_on_policy_runner.py`
- Modify: `rsl_rl/rsl_rl/runners/__init__.py`
- Test: `source/isaaclab_rl/test/test_smp_reward_logging.py`

- [ ] **Step 1: Write the failing SDS-ensemble reward test**

```python
def test_smp_reward_uses_fixed_timestep_ensemble():
    rewarder = SMPReward(num_diffusion_steps=50, timesteps_k=[22, 15, 8], reward_scale=1.0)
    eps = {
        22: torch.zeros(4, 10, 131),
        15: torch.zeros(4, 10, 131),
        8: torch.zeros(4, 10, 131),
    }
    eps_hat = {
        22: torch.ones(4, 10, 131),
        15: torch.ones(4, 10, 131),
        8: torch.ones(4, 10, 131),
    }
    out = rewarder.compute(eps=eps, eps_hat=eps_hat)
    assert out["reward"].shape == (4,)
    assert set(out["per_timestep_mse"].keys()) == {22, 15, 8}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: FAIL with missing `SMPReward`.

- [ ] **Step 3: Implement the rewarder and runner with TensorBoard gap logging**

```python
class SMPReward:
    def __init__(self, num_diffusion_steps, timesteps_k, reward_scale, adaptive_norm_decay=0.99):
        self.timesteps_k = timesteps_k
        self.reward_scale = reward_scale
        self.running_mse = {t: None for t in timesteps_k}

    def compute(self, eps, eps_hat):
        per_timestep_mse = {}
        normed_terms = []
        for t in self.timesteps_k:
            mse = (eps_hat[t] - eps[t]).pow(2).flatten(start_dim=1).mean(dim=1)
            per_timestep_mse[t] = mse
            normed_terms.append(self._normalize(t, mse))
        noise_mse = torch.stack(normed_terms, dim=0).mean(dim=0)
        reward = torch.exp(-self.reward_scale * noise_mse)
        return {"reward": reward, "noise_mse": noise_mse, "per_timestep_mse": per_timestep_mse}
```

- [ ] **Step 4: Validate logging tags from the runner helper**

Run: `python -m pytest source/isaaclab_rl/test/test_smp_reward_logging.py -q`
Expected: PASS.

Run: `python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-SMP-Velocity-Flat-G1-v0 --max_iterations 2 --num_envs 64 --headless`
Expected: training starts, TensorBoard event files are created, and tags include `SMP/noise_mse`, `SMP/reward`, `SMP/t22/noise_mse`, `SMP/t15/noise_mse`, `SMP/t8/noise_mse`, plus histograms `SMP/eps_true_t22`, `SMP/eps_pred_t22`, `SMP/eps_gap_t22`.

- [ ] **Step 5: Commit**

```bash
git add rsl_rl/rsl_rl/diffusion/smp_reward.py \
        rsl_rl/rsl_rl/runners/smp_on_policy_runner.py \
        rsl_rl/rsl_rl/runners/__init__.py \
        source/isaaclab_rl/test/test_smp_reward_logging.py
git commit -m "feat: add frozen-prior SMP reward runner"
```

### Task 6: Register The New G1 SMP Task And Hydra Coverage

**Files:**
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py`
- Modify: `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py`
- Modify: `source/isaaclab_tasks/test/test_hydra.py`

- [ ] **Step 1: Add a failing Hydra coverage test for the new task**

```python
def test_g1_smp_hydra_registration():
    @hydra_task_config_test("Isaac-SMP-Velocity-Flat-G1-v0", "rsl_rl_cfg_entry_point")
    def main(env_cfg, agent_cfg):
        assert agent_cfg.runner_type == "rsl_rl.runners:SMPOnPolicyRunner"
        assert agent_cfg.smp_prior.window_size == 10
        assert agent_cfg.smp_prior.num_diffusion_steps == 50
        assert agent_cfg.smp_prior.timesteps_k == [22, 15, 8]

    main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest source/isaaclab_tasks/test/test_hydra.py -q`
Expected: FAIL because `Isaac-SMP-Velocity-Flat-G1-v0` is not registered or the agent cfg is missing SMP fields.

- [ ] **Step 3: Add `G1SMPRunnerCfg` and register the task**

```python
@configclass
class G1SMPRunnerCfg(SMPRunnerCfg):
    experiment_name = "g1_smp"
    num_steps_per_env = 24
    max_iterations = 10000
    policy = RslRlPpoActorCriticCfg(...)
    algorithm = RslRlPpoAlgorithmCfg(...)
    smp_prior = SMPPriorCfg(
        checkpoint_path="logs/smp_prior/g1/model_latest.pt",
        feature_dim=131,
        window_size=10,
        num_diffusion_steps=50,
        timesteps_k=[22, 15, 8],
        reward_scale=1.0,
    )
```

- [ ] **Step 4: Re-run Hydra coverage**

Run: `python -m pytest source/isaaclab_tasks/test/test_hydra.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/__init__.py \
        source/isaaclab_tasks/test/test_hydra.py
git commit -m "feat: register G1 SMP locomotion task"
```

### Task 7: End-To-End Verification And TensorBoard Acceptance Criteria

**Files:**
- Verify only: `scripts/imitation_learning/smp/train_motion_prior.py`
- Verify only: `scripts/reinforcement_learning/rsl_rl/train.py`
- Verify only: `logs/`

- [ ] **Step 1: Export the real G1 prior dataset**

Run: `python scripts/imitation_learning/smp/export_g1_motion_dataset.py --input source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/motion.npz --output logs/smp_prior/g1/dataset/g1_smp_windows.npz --window-size 10`
Expected: exported dataset exists and contains `frames`, `window_size`, and `feature_dim`.

- [ ] **Step 2: Pretrain the prior until TensorBoard gap metrics are visible**

Run: `python scripts/imitation_learning/smp/train_motion_prior.py --dataset logs/smp_prior/g1/dataset/g1_smp_windows.npz --logdir logs/smp_prior/g1/pretrain --batch-size 256 --num-diffusion-steps 50 --window-size 10`
Expected: checkpoints and TensorBoard event files appear under `logs/smp_prior/g1/pretrain`.

- [ ] **Step 3: Launch a short online PPO smoke run**

Run: `python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-SMP-Velocity-Flat-G1-v0 --num_envs 256 --max_iterations 5 --headless`
Expected: run directory created under `logs/rsl_rl/g1_smp/`, with event files showing both PPO and SMP tags.

- [ ] **Step 4: Inspect TensorBoard tags and confirm the required visualizations exist**

Run: `tensorboard --logdir logs`
Expected:
- pretraining scalars: `SMPPretrain/noise_mse`, `SMPPretrain/t22/noise_mse`, `SMPPretrain/t15/noise_mse`, `SMPPretrain/t8/noise_mse`
- online scalars: `SMP/noise_mse`, `SMP/reward`, `SMP/t22/noise_mse`, `SMP/t15/noise_mse`, `SMP/t8/noise_mse`
- histograms: `SMP/eps_true_t22`, `SMP/eps_pred_t22`, `SMP/eps_gap_t22` (and the same for `t15`, `t8`)
- success signal: the scalar noise gap trends downward over training and the histogram spread of `eps_gap` contracts compared with the first logged steps

- [ ] **Step 5: Commit**

```bash
git add scripts/imitation_learning/smp/export_g1_motion_dataset.py \
        scripts/imitation_learning/smp/train_motion_prior.py \
        source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py \
        rsl_rl/rsl_rl/runners/smp_on_policy_runner.py
git commit -m "test: verify SMP prior training and TensorBoard noise-gap logging"
```

## Notes For The Implementer

- Use the paper's first-pass defaults exactly unless a test forces a deviation: `window_size=10`, `num_diffusion_steps=50`, predict `epsilon`, fixed `K={22,15,8}`, and EMA on the prior.
- Do not make the PPO policy itself a diffusion policy in this implementation. The diffusion model is a frozen prior and reward model, not the action generator.
- Keep the online and offline feature definitions in one shared module so the prior never trains on a representation that the policy cannot produce at runtime.
- Make TensorBoard verification non-optional. The user requirement is not satisfied until the event files contain both scalar gap metrics and histogram views of `eps_hat - eps`.
- Delay CFG, style labels, and generative state initialization until the unconditional path above is stable and verified.
