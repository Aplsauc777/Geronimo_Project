"""This is a hexapod manager-based reinforcement-learning task for Geronimo"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Hexapod-Scratch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HexapodPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Hexapod-Forward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_forward_env_cfg:HexapodForwardEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_forward_ppo_cfg:HexapodForwardPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Hexapod-Terrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_mild_rough_env_cfg:HexapodMildRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_forward_ppo_cfg:HexapodForwardPPORunnerCfg",
    },
)