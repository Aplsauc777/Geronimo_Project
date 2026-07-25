

from __future__ import annotations
import math

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

GERONIMO_USD = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision.usd"

FOOT_BODY_NAMES = [
    "servo_cpy_8",  # front right
    "servo_cpy_10",  # middle right
    "servo_cpy_15",  # back right
    "servo_cpy_5",    # front left
    "servo_cpy_14",  # middle left
    "servo_cpy",  # back left
]

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

JOINTS = (
    FRONT_RIGHT_JOINTS
    + MIDDLE_RIGHT_JOINTS
    + BACK_RIGHT_JOINTS
    + FRONT_LEFT_JOINTS
    + MIDDLE_LEFT_JOINTS
    + BACK_LEFT_JOINTS
)

STANDING_HEIGHT = 0.165818
MINIMUM_STANDING_HEIGHT = 0.55 * STANDING_HEIGHT


GERONIMO_CFG = ArticulationCfg(

    spawn=sim_utils.UsdFileCfg(
        usd_path=GERONIMO_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
           
            enabled_self_collisions=False,

            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
    ),

    actuators={
        "leg_motors": ImplicitActuatorCfg(
            joint_names_expr=JOINTS,

            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness=60.0,
            damping=6.0,
            armature=0.01,
        )
    },
)




@configclass
class HexapodSceneCfg(InteractiveSceneCfg):
    """Defines the simulation world."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = GERONIMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    foot_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/servo_cpy.*",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=3000.0,
        ),
    )




@configclass
class ActionsCfg:
    "These are the actions sent by the policy to the robot"

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINTS,
        scale=0.15,
        use_default_offset=True,
        preserve_order=True,
    )

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )




@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):

      
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

   
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)


        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        velocity_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )

        joint_positions = ObsTerm(func=mdp.joint_pos_rel)

        joint_velocities = ObsTerm(func=mdp.joint_vel_rel)

        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()




@configclass
class EventCfg:

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (0.0, 0.01),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },

            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.03, 0.03),
                "roll": (-0.10, 0.10),
                "pitch": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
            }
        }
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.10, 0.05),
        }
    )




@configclass
class RewardsCfg:
    """The creates the rewards for geronimo"""


    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )

    stay_still_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": 0.25
        }
    )

    stay_still_yaw = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.25,
        }
    )

    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-20.0,
        params={
            "target_height": STANDING_HEIGHT,
        },
    )

    vertical_velocity = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-2.0,
    )

    roll_pitch_velocity = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.5,
    )

    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
    )

    joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    action_magnitude = RewTerm(
        func=mdp.action_l2,
        weight=-0.001,
    )
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )



    # forward_progress = RewTerm(
    #     func=hex_rewards.forward_progress_reward,
    #     weight=6.0,
    #     params={
    #         "target_speed": 0.35,
    #         "direction": (1.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # forward_speed = RewTerm(
    #     func=hex_rewards.track_forward_speed,
    #     weight=2.0,
    #     params={
    #         "target_speed": 0.25,
    #         "direction": (1.0, 0.0),
    #         "std": 0.15,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # standing_still = RewTerm(
    #     func=hex_rewards.standing_still_penalty,
    #     weight=-2.0,
    #     params={
    #         "min_speed": 0.08,
    #         "direction": (1.0, 0.0),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # upright = RewTerm(
    #     func=hex_rewards.upright_reward,
    #     weight=0.8,
    #     params={
    #         "std": 0.5,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # body_height = RewTerm(
    #     func=hex_rewards.body_height_reward,
    #     weight=0.8,
    #     params={
    #         "target_height": 0.30,
    #         "min_height": 0.18,
    #         "std": 0.20,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # action_smoothness = RewTerm(
    #     func=hex_rewards.action_smoothness_penalty,
    #     weight=-0.05,
    # )

    # joint_velocity = RewTerm(
    #     func=hex_rewards.joint_velocity_penalty,
    #     weight=-0.001,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # yaw_spin = RewTerm(
    #     func=hex_rewards.yaw_spin_penalty,
    #     weight=-0.02,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )




@configclass
class TerminationsCfg:
    """Defines when the episode ends."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": MINIMUM_STANDING_HEIGHT,
        },
    )

    tipped_over = DoneTerm(
        func=mdp.bad_orientation,
        params={
            "max_tilt": 0.8,
        },
    )




@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):

    scene: HexapodSceneCfg = HexapodSceneCfg(
        num_envs=4,
        env_spacing=3.0,
        # clone_in_fabric=True,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """these are the final environmental things to configure"""

        self.dim_dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.25)