# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    foot_sensor = env.unwrapped.scene["foot_contact_sensor"]
    print("\nFoot sensor body order:")
    for index, body_name in enumerate(foot_sensor.body_names):
        print(f"{index}: {body_name}")

    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = torch.zeros(
                (
                    env.unwrapped.num_envs,
                    env.unwrapped.action_manager.total_action_dim,
                ),
                device=env.unwrapped.device,
            )
            # apply actions
            env.step(actions)

            if step_count % 30 == 0:
                forces = foot_sensor.data.net_forces_w[0]
                force_magnitudes = torch.linalg.vector_norm(forces, dim=-1)

                print("\nFOOT CONTACT FORCES")

                for body_name, force in zip(foot_sensor.body_names, force_magnitudes):
                    touching = force.item() > 0.5

                    print(
                        f"{body_name:<25} "
                        f"force={force.item():7.3f} N | "
                        f"contact={touching}"
                    )
            step_count += 1

            

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
