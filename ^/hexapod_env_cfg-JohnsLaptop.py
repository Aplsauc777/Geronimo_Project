# ------------------------------------------------------------
# hexapod_env_cfg.py
#
# Purpose:
# This is the main environment configuration.
#
# It defines:
# - the world/scene
# - the robot asset
# - the action space
# - the observation space
# - reset behavior
# - rewards
# - terminations
# - simulation timestep
#
# This is the heart of the RL task.
# ------------------------------------------------------------

from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# Import our MDP package.
# This includes built-in Isaac Lab MDP tools and our custom rewards/terminations.
import isaaclab.envs.mdp as base_mdp
from isaaclab_tasks.manager_based.classic.hexapod_scratch.mdp import rewards as hex_rewards
from isaaclab_tasks.manager_based.classic.hexapod_scratch.mdp import terminations as hex_terminations
# ------------------------------------------------------------
# 1. Robot USD path
# ------------------------------------------------------------

# This is the USD generated from your collision-fixed URDF.
GERONIMO_USD = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision.usd"

FRONT_RIGHT_JOINTS = [
    "revolute_1",
    "revolute_2",
    "revolute_3",
]

MIDDLE_RIGHT_JOINTS = [
    "revolute_1_1",
    "revolute_2_1",
    "revolute_3_1",
]

BACK_RIGHT_JOINTS = [
    "revolute_1_2",
    "revolute_2_2",
    "revolute_3_2",
]

BACK_LEFT_JOINTS = [
    "revolute_1_5",
    "revolute_2_5",
    "revolute_3_5",
]

MIDDLE_LEFT_JOINTS = [
    "revolute_1_3",
    "revolute_2_3",
    "revolute_3_3",
]

FRONT_LEFT_JOINTS = [
    "revolute_1_4",
    "revolute_2_4",
    "revolute_3_4",
]

GERONIMO_MOTOR_JOINTS = (
    FRONT_RIGHT_JOINTS
    + MIDDLE_RIGHT_JOINTS
    + BACK_RIGHT_JOINTS
    + FRONT_LEFT_JOINTS
    + MIDDLE_LEFT_JOINTS
    + BACK_LEFT_JOINTS
)
# ------------------------------------------------------------
# 2. Robot asset config
# ------------------------------------------------------------

GERONIMO_CFG = ArticulationCfg(
    # spawn tells Isaac Lab how to load the robot into the scene.
    spawn=sim_utils.UsdFileCfg(
        usd_path=GERONIMO_USD,

        # Rigid body settings for the robot links.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),

        # Settings for the whole robot articulation.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Start with self-collisions off because imported CAD collisions
            # can overlap and make the robot explode.
            enabled_self_collisions=False,

            # More position iterations usually make joints more stable.
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),

    # Initial pose/state of the robot.
    init_state=ArticulationCfg.InitialStateCfg(
        # Spawn above ground so it does not start intersecting the floor.
        pos=(0.0, 0.0, 1),
    ),

    # Actuators define how the policy can control joints.
    actuators={
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=GERONIMO_MOTOR_JOINTS,

            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness=60.0,
            damping=6.0,
        )
    },
)


# ------------------------------------------------------------
# 3. Scene config
# ------------------------------------------------------------

@configclass
class HexapodSceneCfg(InteractiveSceneCfg):
    """Defines the simulation world."""

    # Ground plane.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # The robot.
    robot: ArticulationCfg = GERONIMO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # Light so we can see the robot.
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=3000.0,
        ),
    )


# ------------------------------------------------------------
# 4. Actions
# ------------------------------------------------------------

@configclass
class ActionsCfg:
    """Defines what the policy controls.

    The policy outputs numbers in roughly [-1, 1].
    JointPositionActionCfg converts those into joint target positions.
    """

    joint_pos = JointPositionActionCfg(
        asset_name="robot",
        joint_names=GERONIMO_MOTOR_JOINTS,
        scale=0.5,
        use_default_offset=True,
    )


# ------------------------------------------------------------
# 5. Observations
# ------------------------------------------------------------

@configclass
class ObservationsCfg:
    """Defines what the policy can see."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations sent to the policy neural network."""

        # Base linear velocity.
        # Helps the robot know how fast it is moving.
        base_lin_vel = ObsTerm(func=base_mdp.base_lin_vel)

        # Base angular velocity.
        # Helps it know if it is spinning or tipping.
        base_ang_vel = ObsTerm(func=base_mdp.base_ang_vel)

        # Gravity direction in body frame.
        # Helps it know if it is upright.
        projected_gravity = ObsTerm(func=base_mdp.projected_gravity)

        # Joint positions relative to default.
        joint_pos = ObsTerm(func=base_mdp.joint_pos_rel)

        # Joint velocities.
        joint_vel = ObsTerm(func=base_mdp.joint_vel_rel)

        # Previous action.
        # Helps smooth control over time.
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            # No observation noise yet.
            self.enable_corruption = False

            # Concatenate all observation terms into one flat vector.
            self.concatenate_terms = True

    # This creates the policy observation group.
    policy: PolicyCfg = PolicyCfg()


# ------------------------------------------------------------
# 6. Reset events
# ------------------------------------------------------------

@configclass
class EventCfg:
    """Defines what happens at reset."""

    reset_joints = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            # Reset all robot joints.
            "asset_cfg": SceneEntityCfg("robot"),

            # Start close to default pose.
            "position_range": (-0.05, 0.05),

            # Start with small random joint velocities.
            "velocity_range": (-0.05, 0.05),
        },
    )


# ------------------------------------------------------------
# 7. Rewards
# ------------------------------------------------------------

@configclass
class RewardsCfg:
    """Defines the learning objective."""

    # Small survival reward.
    # Encourages robot not to terminate instantly.
    alive = RewTerm(
        func=base_mdp.is_alive,
        weight=0.05,
    )

    # Main reward: actual forward movement.
    # Standing still gives 0 from this term

    forward_progress = RewTerm(
        func=hex_rewards.forward_progress_reward,
        weight=6.0,
        params={
            "target_speed": 0.35,
            "direction": (1.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Main walking reward:
    # walk in +X direction at about 0.25 m/s.
    forward_speed = RewTerm(
        func=hex_rewards.track_forward_speed,
        weight=2.0,
        params={
            "target_speed": 0.25,
            "direction": (1.0, 0.0),
            "std": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Penalize standing still.
    standing_still = RewTerm(
        func=hex_rewards.standing_still_penalty,
        weight=-2.0,
        params={
            "min_speed": 0.08,
            "direction": (1.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Keep body upright.
    upright = RewTerm(
        func=hex_rewards.upright_reward,
        weight=0.8,
        params={
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Keep body from scraping the ground.
    body_height = RewTerm(
        func=hex_rewards.body_height_reward,
        weight=0.8,
        params={
            "target_height": 0.30,
            "min_height": 0.18,
            "std": 0.20,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Penalize sudden changes in action.
    action_smoothness = RewTerm(
        func=hex_rewards.action_smoothness_penalty,
        weight=-0.05,
    )

    # Penalize fast joint motion.
    joint_velocity = RewTerm(
        func=hex_rewards.joint_velocity_penalty,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Penalize spinning.
    yaw_spin = RewTerm(
        func=hex_rewards.yaw_spin_penalty,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ------------------------------------------------------------
# 8. Terminations
# ------------------------------------------------------------

@configclass
class TerminationsCfg:
    """Defines when the episode ends."""

    # End after episode_length_s.
    time_out = DoneTerm(
        func=base_mdp.time_out,
        time_out=True,
    )

    # End if body gets too low.
    body_low = DoneTerm(
        func=hex_terminations.body_too_low,
        params={
            "min_height": 0.12,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # End if robot flips/tilts too much.
    bad_orientation = DoneTerm(
        func=hex_terminations.bad_orientation,
        params={
            "max_tilt": 0.8,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


# ------------------------------------------------------------
# 9. Full environment config
# ------------------------------------------------------------

@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """Full RL environment config for Geronimo."""

    # Scene settings.
    scene: HexapodSceneCfg = HexapodSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
        clone_in_fabric=True,
    )

    # MDP settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Final environment settings after config is created."""

        # Action decimation:
        # Policy sends an action every 4 physics steps.
        self.decimation = 4

        # Episode length in seconds.
        self.episode_length_s = 10.0

        # Camera view for debugging.
        self.viewer.eye = (3.0, -4.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.3)

        # Physics timestep.
        self.sim.dt = 1.0 / 120.0

        # Render every control step.
        self.sim.render_interval = self.decimation