

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
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from . import mdp

GERONIMO_USD = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision_heavy.usd"

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
MINIMUM_STANDING_HEIGHT = 0.70 * STANDING_HEIGHT
GAIT_CYCLE_TIME = 1.2


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
        ),
    },
)




@configclass
class HexapodSceneCfg(InteractiveSceneCfg):
    """Defines the simulation world."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    robot: ArticulationCfg = GERONIMO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
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
        scale=0.30,
        use_default_offset=True,
        preserve_order=True,
    )

@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.12, 0.22),
            lin_vel_y=(-0.03, 0.03),
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

        git_phase = ObsTerm(func=mdp.gait_phase_observation, params={"cycle_time": GAIT_CYCLE_TIME})


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
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            }
        }
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.03, 0.03),
            "velocity_range": (-0.1, 0.1),
        }
    )




@configclass
class RewardsCfg:
    """The creates the rewards for geronimo"""



    track_lateral_velocity = RewTerm(
        func=mdp.track_lateral_velocity_exp,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "std": 0.03,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    track_forward_velocity = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": 0.15,
        }
    )

    track_yaw_velocity = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": 0.1,
        }
    )

    tripod_swing_force = RewTerm(
        func=mdp.tripod_swing_force_reward,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "cycle_time": GAIT_CYCLE_TIME,
            "duty_factor": 0.55,
            "force_std": 1.0,
        },
    )

    tripod_stance_velocity = RewTerm(
        func=mdp.tripod_stance_velocity_reward,
        weight=0.2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "cycle_time": GAIT_CYCLE_TIME,
            "duty_factor": 0.55,
            "velocity_std": 0.05,
            "contact_threshold": 1.0,
            "command_threshold": 0.03,
        },
    )

    tripod_stance_contact = RewTerm(
        func=mdp.tripod_stance_contact_reward,
        weight=0.15,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "cycle_time": GAIT_CYCLE_TIME,
            "duty_factor": 0.55,
            "contact_threshold": 1.0,
            "transition_width": 0.2,
            "worst_foot_weight": 0.4,
            "phase_transition_fraction": 0.05,
            "command_threshold": 0.03,
        }
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "threshold": 0.2,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
        },
    )

    # alive = RewTerm(
    #     func=mdp.is_alive,
    #     weight=0.15,
    # )

    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
    )

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={
            "target_height": STANDING_HEIGHT,
        }
    )

    vertical_velocity = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
    )

    roll_pitch_velocity = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.015,
    )

    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-10.0,
    )

    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS, preserve_order=True),
        },
    )





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
            "limit_angle": 0.8,
        },
    )




@configclass
class HexapodForwardEnvCfg(ManagerBasedRLEnvCfg):

    scene: HexapodSceneCfg = HexapodSceneCfg(
        num_envs=1024,
        env_spacing=3.0,
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """these are the final environmental things to configure"""

        self.sim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.25)