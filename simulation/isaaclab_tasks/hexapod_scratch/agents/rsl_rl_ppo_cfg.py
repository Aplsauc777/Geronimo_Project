# ------------------------------------------------------------
# rsl_rl_ppo_cfg.py
#
# Purpose:
# This file configures PPO training for RSL-RL.
#
# PPO is the reinforcement learning algorithm.
# The environment defines the task.
# This file defines how the neural network learns.
# ------------------------------------------------------------

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class HexapodPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for Geronimo hexapod."""

    # Number of steps collected per environment before each PPO update.
    num_steps_per_env = 24

    # Start small while debugging.
    max_iterations = 300

    # Save a checkpoint every 50 iterations.
    save_interval = 50

    # Folder name under logs/rsl_rl.
    experiment_name = "geronimo_hexapod_scratch"

    # Optional run name.
    run_name = ""

    # Neural network settings.
    policy = RslRlPpoActorCriticCfg(
        # Initial exploration noise.
        init_noise_std=1.0,

        # Keep normalization off at first to simplify debugging.
        actor_obs_normalization=False,
        critic_obs_normalization=False,

        # Network size.
        # Bigger is more powerful but slower.
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],

        # ELU is common in Isaac Lab locomotion configs.
        activation="elu",
    )

    # PPO algorithm settings.
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )