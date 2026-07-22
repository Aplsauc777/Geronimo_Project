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
    mount_yaw: float
    signs: tuple[float, float, float]
    offsets: tuple[float, float, float]
    
FRONT_RIGHT_CONFIG = LegConfig(
    
    name="front_right",
    joint_names=tuple(FRONT_RIGHT_LEG),
    mount_yaw=-math.pi / 3.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

MIDDLE_RIGHT_CONFIG = LegConfig(
    name="middle_right",
    joint_names=tuple(MIDDLE_RIGHT_LEG),
    mount_yaw=-math.pi / 2.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

BACK_RIGHT_CONFIG = LegConfig(
    name="back_right",
    joint_names=tuple(BACK_RIGHT_LEG),
    mount_yaw=-2 * math.pi / 3.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

FRONT_LEFT_CONFIG = LegConfig(
    
    name="front_left",
    joint_names=tuple(FRONT_LEFT_LEG),
    mount_yaw=math.pi / 3.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

MIDDLE_LEFT_CONFIG = LegConfig(
    name="middle_left",
    joint_names=tuple(MIDDLE_LEFT_LEG),
    mount_yaw=math.pi / 2.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

BACK_LEFT_CONFIG = LegConfig(
    name="back_left",
    joint_names=tuple(BACK_LEFT_LEG),
    mount_yaw=2 * math.pi / 3.0,
    signs=(1.0, 1.0, -1.0),
    offsets=(0.0, 0.0, 0.0),
)

LEGS = (
    FRONT_RIGHT_CONFIG,
    MIDDLE_RIGHT_CONFIG,
    BACK_RIGHT_CONFIG,
    FRONT_LEFT_CONFIG,  
    MIDDLE_LEFT_CONFIG,
    BACK_LEFT_CONFIG,
)



GERONIMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GERONIMO_PATH,

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
        joint_vel={".*": 0.0},
    ),

    actuators={
        "leg_motors": ImplicitActuatorCfg(

            joint_names_expr=ALL_LEGS,

            effort_limit_sim=300.0,

            velocity_limit_sim=200.0,

            stiffness=300.0,

            damping=35.0,

            armature=0.02,
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
                static_friction=2.0,
                dynamic_friction=1.5,
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



class HexapodIK:
    
    def __init__(self, legs):
        self.legs = {leg.name: leg for leg in legs}

    def smoothstep(self, value: float) -> float:
        return value * value * (3.0 - 2.0 * value)
    
    def foot_trajectory(self, phase: float, ground_time: float, step_length: float, step_height: float) -> tuple[float, float, float]:
        phase = phase % 1.0
        if phase < ground_time:
            progress = self.smoothstep(phase / ground_time)
            dx = (step_length / 2.0 - step_length * progress)
            dy = 0.0
            dz = 0.0
            state = "ground"
        else:
            progress = (phase - ground_time) / (1.0 - ground_time)
            horiz = self.smoothstep(progress)
            dx = (-step_length / 2.0 + step_length * horiz)
            dy = 0.0
            dz = (0.5 * step_height * (1.0 - math.cos(2.0 * math.pi * progress)))
            state = "air"
        return dx, dy, dz, state

    def solve_leg_local(self, xl, yl, zl, knee_down=True) -> tuple[float, float, float]:

        s1 = math.atan2(yl, xl)

        horizontal_distance = math.hypot(xl, yl) - COXA_LENGTH
        d = math.hypot(horizontal_distance, zl)
        reach = FEMUR_LENGTH + TIBIA_LENGTH
        
        if d > reach:
            d = reach - 1e-6

        a = math.atan2(zl, horizontal_distance)
        b = (FEMUR_LENGTH**2 + d**2 - TIBIA_LENGTH**2) / (2 * FEMUR_LENGTH * d)
        b = math.acos(max(-1.0, min(1.0, b)))
        k = (FEMUR_LENGTH**2 + TIBIA_LENGTH**2 - d**2) / (2 * FEMUR_LENGTH * TIBIA_LENGTH)
        k = math.acos(max(-1.0, min(1.0, k)))
        s2 = a + b if knee_down else a - b
        s3 = k - math.pi
        
        return s1, s2, s3
    

    def fk_local(self, s1, s2, s3) -> tuple[float, float, float]:
        radial = COXA_LENGTH + FEMUR_LENGTH * math.cos(s2) + TIBIA_LENGTH * math.cos(s2 + s3)
        vertical = FEMUR_LENGTH * math.sin(s2) + TIBIA_LENGTH * math.sin(s2 + s3)
        return radial * math.cos(s1), radial * math.sin(s1), vertical
    
    def body_vec_to_leg(self, leg, vx, vy, vz) -> tuple[float, float, float]:
        c, s = math.cos(leg.mount_yaw), math.sin(leg.mount_yaw)
        return c * vx + s * vy, -s * vx + c * vy, vz

    def home_geo_from_command(self, leg, q_cmd):
        return tuple((q_cmd[i] - leg.offsets[i]) / leg.signs[i] for i in range(3))

    def command_from_body_offset(self, leg, home_leg_xyz, dx, dy, dz, knee_down=True):
        lx, ly, lz = self.body_vec_to_leg(leg, dx, dy, dz)
        target = (home_leg_xyz[0] + lx, home_leg_xyz[1] + ly, home_leg_xyz[2] + lz)
        s1, s2, s3 = self.solve_leg_local(*target, knee_down=knee_down)
        geo = (s1, s2, s3)
        return tuple(leg.signs[i] * geo[i] + leg.offsets[i] for i in range(3))
    






def reset_scene(scene: InteractiveScene) -> None:

    scene.reset()

def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
) -> None:
    
    geronimo = scene["geronimo"]

    sim_dt = sim.get_physics_dt()

    ik = HexapodIK(LEGS)

    default_pose = geronimo.data.default_joint_pos.clone()

    leg_ids = {}
    leg_home = {}
    for leg in LEGS:
        ids, names = geronimo.find_joints(leg.joint_names, preserve_order=True)
        if len(ids) != 3:
            raise RuntimeError(f"missing servos for {leg.name}. Fount: {names}")
        leg_ids[leg.name] = ids
        q_cmd = default_pose[0, ids].tolist()
        geo_home = ik.home_geo_from_command(leg, q_cmd)
        leg_home[leg.name] = ik.fk_local(*geo_home)

    TRIPOD_A = {
        "front_right",
        "middle_left",
        "back_right",
    }
    TRIPOD_B = {
        "front_left",
        "middle_right",
        "back_left",
    }
    GROUND_TIME = 0.51
    # AXIS = (1.0, 0.0, 0.0)
    # AMPLITUDE = 0.05
    START_DELAY = 0.5
    RAMP_TIME = 3.0
    STEP_LENGTH = 0.3
    STEP_HEIGHT = 0.04
    CYCLE_TIME = 1.0
    print_interval = max(1, int(1.0 / sim_dt))
    step_count = 0

    while simulation_app.is_running():
        joint_target = default_pose.clone()

        total_time = step_count * sim_dt

        if total_time < START_DELAY:
            global_phase = 0.0
            ramp = 0.0
        else:
            gait_time = total_time - START_DELAY
            global_phase = gait_time / CYCLE_TIME % 1.0
            ramp_progress = min(1.0, gait_time / RAMP_TIME)
            ramp = ik.smoothstep(ramp_progress)
       
        #     t = (step_count - wait_steps) * sim_dt
        #     interpolation = math.sin(2.0 * math.pi * t / cycle_time)
        # dx = interpolation * AMPLITUDE * AXIS[0]
        # dy = interpolation * AMPLITUDE * AXIS[1]
        # dz = interpolation * AMPLITUDE * AXIS[2]
        tripod_states = {}

        for leg in LEGS:
            if leg.name in TRIPOD_A:
                leg_phase = global_phase
            else:
                leg_phase = (global_phase + 0.5) % 1.0
            
            dx, dy, dz, gait_state = ik.foot_trajectory(leg_phase, GROUND_TIME, STEP_LENGTH, STEP_HEIGHT)
            tripod_states[leg.name] = gait_state
            dx *= ramp
            dy *= ramp
            dz *= ramp
            cmd = ik.command_from_body_offset(leg, leg_home[leg.name], dx, dy, dz, knee_down=True)

            t_cmd = torch.tensor(cmd, device=default_pose.device, dtype=default_pose.dtype)
            t_cmd = t_cmd.repeat(default_pose.shape[0], 1)
            joint_target[:, leg_ids[leg.name]] = t_cmd
            # cmd = ik.command_from_body_offset(leg, leg_home[leg.name], dx, dy, dz)
            # t_cmd = torch.tensor(cmd, device=default_pose.device, dtype=default_pose.dtype)
            # joint_target[:, leg_ids[leg.name]] = t_cmd.unsqueeze(0).repeat(default_pose.shape[0], 1)

        geronimo.set_joint_position_target(joint_target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        if step_count % print_interval == 0:
            print(f"body offset: ({dx:+.3f}, {dy:+.3f}, {dz:+.3f}) m")

def main() -> None:

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