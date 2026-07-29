"""This is the RSL-RL PPO config for Geronimo"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class HexapodForwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """here is where I define the PPO config for geronimo"""

    num_steps_per_env = 36
    max_iterations = 1500
    save_interval = 50
    experiment_name = "geronimo_forward_ppo"
    run_name = ""
    empirical_normalization = False
    obs_groups = {
        "actor": ["policy"],
        "critic": ["policy"],
    }

    clip_actions = 1.0
    check_for_nan = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )