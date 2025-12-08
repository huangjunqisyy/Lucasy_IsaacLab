# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to manually control a robot with WASD keys."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import numpy as np
import torch

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Manually control a robot with WASD keys.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True 

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import math
import numpy as np

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


class ManualController:
    """Manual controller for robot using WASD keys."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self.linear_velocity = torch.zeros(3, device=device)
        self.angular_velocity = torch.zeros(3, device=device)
        self.max_linear_vel = 1.0
        self.max_angular_vel = 1.0
        
    def get_velocity_command(self):
        """Return SE2 command [vx, vy, wz] in robot base frame."""
        return torch.tensor(
            [self.linear_velocity[0].item(), self.linear_velocity[1].item(), self.angular_velocity[2].item()],
            device=self.device,
            dtype=torch.float32,
        )
    
    def update_from_keys(self, keys_pressed):
        """Update velocity based on pressed keys."""
        # Reset velocities
        self.linear_velocity.zero_()
        self.angular_velocity.zero_()
        
        # WASD control
        if 'w' in keys_pressed:
            self.linear_velocity[0] = self.max_linear_vel  # Forward
        if 's' in keys_pressed:
            self.linear_velocity[0] = -self.max_linear_vel  # Backward
        if 'a' in keys_pressed:
            self.linear_velocity[1] = self.max_linear_vel  # Left
        if 'd' in keys_pressed:
            self.linear_velocity[1] = -self.max_linear_vel  # Right
            
        # QE for rotation
        if 'q' in keys_pressed:
            self.angular_velocity[2] = self.max_angular_vel  # Turn left
        if 'e' in keys_pressed:
            self.angular_velocity[2] = -self.max_angular_vel  # Turn right


class CameraController:
    """Camera controller to follow the robot (no external deps)."""

    def __init__(self, env, camera_distance=5.0, camera_height=2.0):
        self.env = env
        self.camera_distance = camera_distance
        self.camera_height = camera_height

    def update_camera_view(self):
        """Update camera to follow the robot."""
        # Get robot position and yaw heading from quaternion
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        q = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()  # [w, x, y, z]
        w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # Compute forward vector in world from yaw (approx):
        # forward in local = [1, 0, 0]
        # world forward xz using quaternion (ignore roll/pitch for camera)
        # yaw from quaternion
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        # camera behind the robot along -forward
        camera_eye = robot_pos + (-forward * self.camera_distance)
        camera_eye[2] += self.camera_height
        # look at the robot center slightly above
        camera_target = robot_pos.copy()
        camera_target[2] += 0.5
        self.env.unwrapped.sim.set_camera_view(eye=camera_eye, target=camera_target, camera_prim_path="/OmniverseKit_Persp")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with manual control."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        try:
            policy_nn = ppo_runner.alg.actor_critic
        except AttributeError:
            # fallback to policy if actor_critic doesn't exist
            policy_nn = ppo_runner.alg.policy

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=policy_nn.actor_obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=policy_nn.actor_obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # Initialize manual controller and camera controller
    manual_controller = ManualController(device=env.unwrapped.device)
    camera_controller = CameraController(env)
    
    # Get keyboard input handler
    # try:
    #     import keyboard
    #     print("[INFO] Keyboard input enabled. Use WASD/QE to control the robot (ESC to exit)")
    # except ImportError:
    #     print("[WARNING] keyboard module not available. Install with: pip install keyboard")
    #     print("[INFO] Using default policy control instead.")
    #     keyboard = None
    # except Exception as e:
    #     print(f"[ERROR] Keyboard initialization failed: {e}")
    #     print("[INFO] Using default policy control instead.")
    #     keyboard = None

    try:
        from pynput.keyboard import Key, Listener
        print("[INFO] Keyboard input enabled. Use WASD/QE to control the robot (ESC to exit)")
        keyboard_available = True
        current_keys = set()

        def on_press(key):
            try:
                if key.char in ['w', 'a', 's', 'd', 'q', 'e']:
                    current_keys.add(key.char)
            except AttributeError:
                if key == Key.esc:
                    current_keys.add('esc')

        def on_release(key):
            try:
                if key.char in ['w', 's', 'a', 'd', 'q', 'e']:
                    current_keys.discard(key.char)
            except AttributeError:
                if key == Key.esc and 'esc' in current_keys:
                    current_keys.discard('esc')

        listener = Listener(on_press=on_press, on_release=on_release)
        listener.start()
    except ImportError:
        print("[WARNING] pynput module not available. Install with: pip install pynput")
        print("[INFO] Using default policy control instead.")
        keyboard_available = False
    except Exception as e:
        print(f"[ERROR] Keyboard initialization failed: {e}")
        print("[INFO] Using default policy control instead.")
        keyboard_available = False

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        
        # # Check for exit
        # if keyboard and keyboard.is_pressed('esc'):
        #     print("[INFO] ESC pressed, exiting...")
        #     break

        # Check for exit
        if keyboard_available and 'esc' in current_keys:
            print("[INFO] ESC pressed, exiting...")
            listener.stop()
            break
            
        # # Get current key presses
        # keys_pressed = []
        # if keyboard:
        #     if keyboard.is_pressed('w'):
        #         keys_pressed.append('w')
        #     if keyboard.is_pressed('s'):
        #         keys_pressed.append('s')
        #     if keyboard.is_pressed('a'):
        #         keys_pressed.append('a')
        #     if keyboard.is_pressed('d'):
        #         keys_pressed.append('d')
        #     if keyboard.is_pressed('q'):
        #         keys_pressed.append('q')
        #     if keyboard.is_pressed('e'):
        #         keys_pressed.append('e')

        # Get current key presses (获取当前按下的键)
        keys_pressed = []
        if keyboard_available:
            # 从当前按键集合中提取有效键
            for key in ['w', 's', 'a', 'd', 'q', 'e']:
                if key in current_keys:
                    keys_pressed.append(key)
        
        # # Update manual controller
        # if keys_pressed:
        #     manual_controller.update_from_keys(keys_pressed)
        #     # Override the command in the environment (expects [vx, vy, wz])
        #     velocity_cmd = manual_controller.get_velocity_command()
        #     cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
        #     # broadcast to all envs
        #     cmd_term.vel_command_b[:] = velocity_cmd.unsqueeze(0).repeat(env.num_envs, 1)
        #     # ensure angular velocity mode (not heading) and not standing
        #     if hasattr(cmd_term, "is_heading_env"):
        #         cmd_term.is_heading_env[:] = False
        #     if hasattr(cmd_term, "is_standing_env"):
        #         cmd_term.is_standing_env[:] = False

        # Update manual controller (原有逻辑不变)
        if keys_pressed:
            manual_controller.update_from_keys(keys_pressed)
            # Override the command in the environment (expects [vx, vy, wz])
            velocity_cmd = manual_controller.get_velocity_command()
            cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
            # broadcast to all envs
            cmd_term.vel_command_b[:] = velocity_cmd.unsqueeze(0).repeat(env.num_envs, 1)
            # ensure angular velocity mode (not heading) and not standing
            if hasattr(cmd_term, "is_heading_env"):
                cmd_term.is_heading_env[:] = False
            if hasattr(cmd_term, "is_standing_env"):
                cmd_term.is_standing_env[:] = False
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            
        # Update camera to follow robot
        camera_controller.update_camera_view()
            
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function (hydra wrapper injects configs)
    main()
    # close sim app
    simulation_app.close()
