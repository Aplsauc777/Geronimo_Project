# ------------------------------------------------------------
# hexapod_scratch/__init__.py
#
# Purpose:
# This file registers our custom hexapod RL task with Gymnasium.
#
# After this file is imported, Isaac Lab will know this task name:
#
#   Isaac-Hexapod-Geronimo-Scratch-v0
#
# Then we can run:
#
#   isaaclab.bat -p scripts\random_agent.py --task Isaac-Hexapod-Geronimo-Scratch-v0
# ------------------------------------------------------------

import gymnasium as gym

# Import the agents package so we can point to the PPO config.
from . import agents


# Gymnasium registration tells Isaac Lab:
# 1. What the task is called.
# 2. What environment class to use.
# 3. What config file defines the task.
# 4. What agent config to use for training.
#
# Isaac Lab's manager-based tasks use entry_point="isaaclab.envs:ManagerBasedRLEnv".
gym.register(
    id="Isaac-Hexapod-Geronimo-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # This points to the environment config class we will create.
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodEnvCfg",

        # This points to the RSL-RL PPO training config we will create.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodPPORunnerCfg",
    },
)