# ------------------------------------------------------------
# check_hexapod_task.py
#
# Purpose:
# Check whether our custom hexapod task is registered.
#
# Important:
# Isaac Lab needs Isaac Sim/Kit started before importing some
# modules that depend on pxr/USD.
# ------------------------------------------------------------

import argparse

from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 1. Start Isaac Sim / Isaac Lab app first
# ------------------------------------------------------------

parser = argparse.ArgumentParser(description="Check custom Hexapod task registration.")

# Adds Isaac Lab arguments like --headless and --device.
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Launch Isaac Sim.
# Do this BEFORE importing isaaclab_tasks.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ------------------------------------------------------------
# 2. Now import Gymnasium and Isaac Lab tasks
# ------------------------------------------------------------

import gymnasium as gym
import isaaclab_tasks


# ------------------------------------------------------------
# 3. Search registered tasks
# ------------------------------------------------------------

all_task_names = list(gym.registry.keys())

hexapod_tasks = [
    name for name in all_task_names
    if "Hexapod" in name or "Geronimo" in name
]

print("\nRegistered Hexapod/Geronimo tasks:")

if len(hexapod_tasks) == 0:
    print("None found.")
else:
    for task_name in hexapod_tasks:
        print(task_name)


# ------------------------------------------------------------
# 4. Close Isaac Sim cleanly
# ------------------------------------------------------------

simulation_app.close()