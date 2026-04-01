# Motor Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real-time motor observer window to the C++ MuJoCo simulator that lets the user switch motors during runtime and view low-level aligned input/output curves.

**Architecture:** Capture motor command and feedback signals at the MuJoCo bridge boundary, store them in a small ring buffer, and render a dedicated GLFW sidecar window using MuJoCo-native plotting primitives. Keep plotting isolated from physics and bridge code so signal capture, state/history, and rendering remain independently testable.

**Tech Stack:** C++17, MuJoCo `mjvFigure`/`mjr_figure`, GLFW, existing `unitree_mujoco` bridge/threading.

---

### Task 1: Add testable motor-scope state and signal formatting primitives

**Files:**
- Create: `/home/lucas/unitree_rl/unitree_mujoco/simulate/src/motor_scope.h`
- Create: `/home/lucas/unitree_rl/unitree_mujoco/simulate/tests/test_motor_scope.cc`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails because `motor_scope.h` does not exist**
- [ ] **Step 3: Implement minimal ring-buffer, selected-motor state, and signal snapshot structs**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Capture aligned low-level motor signals in the bridge

**Files:**
- Modify: `/home/lucas/unitree_rl/unitree_mujoco/simulate/src/unitree_sdk2_bridge.h`
- Modify: `/home/lucas/unitree_rl/unitree_mujoco/simulate/src/motor_scope.h`
- Test: `/home/lucas/unitree_rl/unitree_mujoco/simulate/tests/test_motor_scope.cc`

- [ ] **Step 1: Extend the failing test to check signal snapshot packing and selected motor switching**
- [ ] **Step 2: Run test to verify it fails for missing capture helpers**
- [ ] **Step 3: Implement bridge-side capture of `q_des`, `dq_des`, `kp`, `kd`, `tau_ff`, `ctrl_applied`, `q`, `dq`, `tau_est`**
- [ ] **Step 4: Run test to verify it passes**

### Task 3: Add the sidecar GLFW scope window and runtime motor switching

**Files:**
- Modify: `/home/lucas/unitree_rl/unitree_mujoco/simulate/src/main.cc`
- Modify: `/home/lucas/unitree_rl/unitree_mujoco/simulate/CMakeLists.txt`
- Modify: `/home/lucas/unitree_rl/unitree_mujoco/simulate/src/motor_scope.h`

- [ ] **Step 1: Add a failing test or compile-time check target if needed for window-facing interfaces**
- [ ] **Step 2: Implement a separate observer window that renders four curve groups for the selected motor**
- [ ] **Step 3: Bind runtime switching keys and on-screen labels for current motor index/name**
- [ ] **Step 4: Build `unitree_mujoco` and verify the new window opens without breaking the main viewer**

### Task 4: Verify end-to-end behavior in the actual simulator

**Files:**
- Verify only

- [ ] **Step 1: Rebuild the simulator**
- [ ] **Step 2: Launch the G1 29DoF scene**
- [ ] **Step 3: Confirm the scope window updates in real time and motor switching works during runtime**
- [ ] **Step 4: Document the runtime keys in the final handoff**
