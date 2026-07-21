import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tripod Gait with Inverse Kinematics Test")

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="# of environments",
)

AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from dataclasses import dataclass
import math
import torch
import numpy as np

GERONIMO_PATH = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision_heavy.usd"

FRONT_RIGHT_LEG = [
    "revolute_1",
    "revolute_2",
    "revolute_3",
]

MIDDLE_RIGHT_LEG = [
    "revolute_1_1",
    "revolute_2_1",
    "revolute_3_1",
]

BACK_RIGHT_LEG = [
    "revolute_1_2",
    "revolute_2_2",
    "revolute_3_2",
]

FRONT_LEFT_LEG = [
    "revolute_1_4",
    "revolute_2_4",
    "revolute_3_4",
]

MIDDLE_LEFT_LEG = [
    "revolute_1_3",
    "revolute_2_3",
    "revolute_3_3",
]

BACK_LEFT_LEG = [
    "revolute_1_5",
    "revolute_2_5",
    "revolute_3_5",
]

ALL_LEGS = (
    FRONT_RIGHT_LEG
    + MIDDLE_RIGHT_LEG
    + BACK_RIGHT_LEG
    + FRONT_LEFT_LEG
    + MIDDLE_LEFT_LEG
    + BACK_LEFT_LEG
)

GAIT_A = [
    FRONT_RIGHT_LEG,
    MIDDLE_LEFT_LEG,
    BACK_RIGHT_LEG,
]

GAIT_B = [
    FRONT_LEFT_LEG,
    MIDDLE_RIGHT_LEG,
    BACK_LEFT_LEG,
]

# fix these values

COXA_LENGTH = 0.0690507
FEMUR_LENGTH = 0.11
TIBIA_LENGTH = 0.2236

HOME_RADIUS = 0.24675914
HOME_HEIGHT = 0.11912510



@dataclass(frozen=True)
class LegConfig:
    name: str
    joint_names: tuple[str, str, str]
    home_xyz: tuple[float, float, float]
    mount_xyz: tuple[float, float, float]
    mount_yaw: float
    signs: tuple[float, float, float]
    offsets: tuple[float, float, float]
    
GERONIMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GERONIMO_PATH,

        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,

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
        joint_vel={".*": 0.0},
    ),

    actuators={
        "leg_motors": ImplicitActuatorCfg(

            joint_names_expr=ALL_LEGS,

            effort_limit_sim=300.0,

            velocity_limit_sim=200.0,

            stiffness=300.0,

            damping=35.0,

            armature=0.01,
        )
    },
)

@configclass
class GeronimoSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),

            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.5,
                dynamic_friction=2.0,
                restitution=0.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    geronimo = GERONIMO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
# @configclass
# class GeronimoSceneCfg(InteractiveSceneCfg):

#     ground = AssetBaseCfg(
#         prim_path="/World/ground",
#         spawn=sim_utils.GroundPlaneCfg(
#             size=(100.0, 100.0),

#             physics_material=sim_utils.RigidBodyMaterialCfg(
#                 static_friction=2.5,
#                 dynamic_friction=2.0,
#                 restitution=0.0,
#             ),
#         ),
#     )

#     light = AssetBaseCfg(
#         prim_path="/World/DomeLight",
#         spawn=sim_utils.DomeLightCfg(
#             intensity=3000.0,
#         ),
#     )

#     geronimo = ArticulationCfg(
#         prim_path="{ENV_REGEX_NS}/geronimo",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=GERONIMO_PATH,

#             rigid_props=sim_utils.RigidBodyPropertiesCfg(
#                 disable_gravity=False,
#                 max_depenetration_velocity=5.0,
#             ),

#             articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#                 enabled_self_collisions=False,
#                 solver_position_iteration_count=8,
#                 solver_velocity_iteration_count=1,
#             ),
#         ),

#         init_state=ArticulationCfg.InitialStateCfg(
#             pos=(0.0, 0.0, 0.5),
#         ),

#         actuators = {
#             "leg_motors": ImplicitActuatorCfg(
#             joint_names_expr=ALL_LEGS,
#             effort_limit_sim=300.0,
#             velocity_limit_sim=200.0,
#             stiffness=50.0,
#             damping=30.0,
#             armature=0.01,
#             )
#         }
#     )

MIDDLE_RIGHT_CONFIG = LegConfig(
    name="middle_right",
    joint_names=tuple(MIDDLE_RIGHT_LEG),
    home_xyz=(HOME_RADIUS, 0.0, -HOME_HEIGHT),
    mount_xyz=(-0.2, 0.0, 0.0),
    mount_yaw=-math.pi / 2.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

LEGS = (
    MIDDLE_RIGHT_CONFIG,
)

class HexapodIK:
    
    def __init__(self, legs):
        self.legs = {leg.name: leg for leg in legs}

    def body_to_leg_frame(self, leg, x, y, z):
        dx = x - leg.mount_xyz[0]
        dy = y - leg.mount_xyz[1]
        dz = z - leg.mount_xyz[2]

        cos, sin = math.cos(leg.mount_yaw), math.sin(leg.mount_yaw)

        x_leg = cos * dx + sin * dy
        y_leg = -sin * dx + cos * dy
        z_leg = dz
        
        return x_leg, y_leg, z_leg

    def solve_leg_local(self, xl, yl, zl, knee_down=True) -> tuple[float, float, float]:

        s1 = math.atan2(yl, xl)

        horizontal_distance = math.hypot(xl, yl) - COXA_LENGTH
        d = math.hypot(horizontal_distance, zl)
        reach = FEMUR_LENGTH + TIBIA_LENGTH
        
        if d > reach:
            d = reach - 1e-6

        a = math.atan2(zl, horizontal_distance)
        b = (FEMUR_LENGTH**2 + d**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * d)
        b = math.acos(np.clip(b, -1.0, 1.0))
        s3 = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - d**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
        s3 = math.acos(np.clip(s3, -1.0, 1.0))
        s3 = s3 - math.pi
        if knee_down:
            s2 = a + b
        else:
            s2 = a - b
        
        return s1, s2, s3
    
    def solve_leg_offset(self, leg_name, x, y, z, knee_down: bool=True) -> tuple[float, float, float]:
        leg = self.legs[leg_name]
        home_x, home_y, home_z = leg.home_xyz
        target_x = home_x + x
        target_y = home_y + y
        target_z = home_z + z
        return self.solve_leg_local(xl=target_x, yl=target_y, zl=target_z, knee_down=knee_down)

    
    def to_command(self, leg_name, s1, s2, s3) -> tuple[float, float, float]:
        leg = self.legs[leg_name]
        angles = s1, s2, s3
        return tuple(leg.signs[i] * angles[i] + leg.offsets[i] for i in range(3))
    
    # This is a test function
    def forward_kinematics(self, leg_name, s1, s2, s3) -> tuple[float, float, float]:
        leg = self.legs[leg_name]
        radial = COXA_LENGTH + FEMUR_LENGTH * math.cos(s2) + TIBIA_LENGTH * math.cos(s2 + s3)
        vertical = FEMUR_LENGTH * math.sin(s2) + TIBIA_LENGTH * math.sin(s2 + s3)
        xl = radial * math.cos(s1)
        yl = radial * math.sin(s1)
        zl = vertical
        cos, sin = math.cos(leg.mount_yaw), math.sin(leg.mount_yaw)
        x = cos * xl - sin * yl + leg.mount_xyz[0]
        y = sin * xl + cos * yl + leg.mount_xyz[1]
        z = zl + leg.mount_xyz[2]
        return x, y, z
    
    def body_offset_to_leg_offset(
            self,
            leg_name: str,
            x_body: float,
            y_body: float,
            z_body: float,
    ) -> tuple[float, float, float]:
        leg = self.legs[leg_name]

        cos, sin = math.cos(leg.mount_yaw), math.sin(leg.mount_yaw)

        x_leg = cos * x_body + sin * y_body
        y_leg = -sin * x_body + cos * y_body
        z_leg = z_body
        return x_leg, y_leg, z_leg
    






def reset_scene(scene: InteractiveScene) -> None:

    scene.reset()

def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
) -> None:
    
    geronimo = scene["geronimo"]

    sim_dt = sim.get_physics_dt()

    ik = HexapodIK(LEGS)

    standing_pose = geronimo.data.default_joint_pos.clone()

    test_leg_cfg = MIDDLE_RIGHT_CONFIG

    test_leg_joint_ids, matched_joint_names = geronimo.find_joints(
        test_leg_cfg.joint_names,
        preserve_order=True,
    )

    if len(test_leg_joint_ids) != 3:
        raise RuntimeError(f"missing joints for {test_leg_cfg.name}")
    print(f"{matched_joint_names}")

    standing_leg_pose = standing_pose[:, test_leg_joint_ids,].clone()
    standing_angles = standing_leg_pose[0].tolist()


    home_ik_angles = ik.solve_leg_offset(
        leg_name=test_leg_cfg.name,
        x=0.0,
        y=0.0,
        z=0.0,
        knee_down=True,
    )

    calibrated_offsets = tuple(
        standing_angles[i] - test_leg_cfg.signs[i] * home_ik_angles[i] for i in range(3)
    )

    body_test_offset = (0.0, 0.0, 0.3,)
    leg_test_offset = ik.body_offset_to_leg_offset(
        leg_name=test_leg_cfg.name,
        x_body=body_test_offset[0],
        y_body=body_test_offset[1],
        z_body=body_test_offset[2],
    )

    final_ik_angles = ik.solve_leg_offset(
        leg_name=test_leg_cfg.name,
        x=leg_test_offset[0],
        y=leg_test_offset[1],
        z=leg_test_offset[2],
        knee_down=True,
    )

    final_command_angles = tuple(
        test_leg_cfg.signs[i] * final_ik_angles[i] + calibrated_offsets[i] for i in range(3)
    )

    maximum_change = max(
        abs(final_command_angles[i] - standing_angles[i]) for i in range(3)
    )


    wait_time = 2.0
    cycle_time = 4.0

    wait_steps = int(wait_time / sim_dt)
    print_interval = max(1, int(1.0 / sim_dt))
    step_count = 0


    while simulation_app.is_running():

        joint_target = standing_pose.clone()

        if step_count < wait_steps:
            interpolation = 0.0

        else:
            elapsed_time = (
                step_count - wait_steps
            ) * sim_dt

            phase = (
                2.0
                * math.pi
                * elapsed_time
                / cycle_time
            )

            interpolation = 0.5 * (
                1.0 - math.cos(phase)
            )
        current_offset = (
            interpolation * leg_test_offset[0],
            interpolation * leg_test_offset[1],
            interpolation * leg_test_offset[2],
        )
        
        current_ik_angles = ik.solve_leg_offset(
            leg_name=test_leg_cfg.name,
            x=current_offset[0],
            y=current_offset[1],
            z=current_offset[2],
            knee_down=True,
        )

        current_command_angle = tuple(
            test_leg_cfg.signs[i] * current_ik_angles[i] + calibrated_offsets[i] for i in range(3)
        )

        current_test_leg_target = torch.tensor(
            current_command_angle,
            device=standing_pose.device,
            dtype=standing_pose.dtype,
        ).unsqueeze(0)

        current_test_leg_target = current_test_leg_target.repeat(
            standing_pose.shape[0],
            1,
        )

        joint_target[
            :,
            test_leg_joint_ids,
        ] = current_test_leg_target

        geronimo.set_joint_position_target(
            joint_target
        )

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        step_count += 1

        if step_count % print_interval == 0:
            actual_angles = geronimo.data.joint_pos[
                0,
                test_leg_joint_ids,
            ]

            print(
                f"blend={interpolation:.3f} | "
                f"target={current_test_leg_target[0].tolist()} | "
                f"actual={actual_angles.tolist()}"
            )

def main() -> None:
    print("[MAIN] Creating simulation context", flush=True)

    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1.0 / 120.0,
        render_interval=2,
    )

    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view(
        eye=(2.5, -3.0, 2.0),
        target=(0.0, 0.0, 0.5),
    )

    print("[MAIN] Creating scene", flush=True)

    scene_cfg = GeronimoSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
    )

    scene = InteractiveScene(scene_cfg)

    print("[MAIN] Resetting simulation", flush=True)

    sim.reset()
    scene.update(sim.get_physics_dt())

    print("[MAIN] Entering run_simulator", flush=True)

    run_simulator(
        sim=sim,
        scene=scene,
    )

    print("[MAIN] run_simulator ended", flush=True)

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()