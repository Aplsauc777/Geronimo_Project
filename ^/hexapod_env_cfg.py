

from __future__ import annotations
import math

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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
    "tibia_moment_v3_1",  # front right
    "tibia_moment_v3_4",  # middle right
    "tibia_moment_v3_2",  # back right
    "tibia_moment_v3",    # front left
    "tibia_moment_v3_5",  # middle left
    "tibia_moment_v3_3",  # back left
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


GERONIMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GERONIMO_USD,

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
        pos=(0.0, 0.0, 1),
    ),

    actuators={
        "leg_motors": mdp.ImplicitActuatorCfg(
            joint_names_expr=JOINTS,

            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness=60.0,
            damping=6.0,
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

    FOOT_CONTACT_SENSOR = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/tibia_moment_v3.*",
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
        scale=math.pi / 2,
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

    reset_joints = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )




@configclass
class RewardsCfg:
    """The creates the rewards for geronimo"""


    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-2.0,
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
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

    # body_low = DoneTerm(
    #     func=hex_terminations.body_too_low,
    #     params={
    #         "min_height": 0.12,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # bad_orientation = DoneTerm(
    #     func=hex_terminations.bad_orientation,
    #     params={
    #         "max_tilt": 0.8,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )




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

        self.dim.dt = 1.0 / 120.0
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.25)