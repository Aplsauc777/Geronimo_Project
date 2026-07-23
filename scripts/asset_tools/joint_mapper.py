# ------------------------------------------------------------
# joint_mapper.py
#
# Purpose:
# This script loads Geronimo and wiggles ONE joint at a time.
#
# Why:
# Your Onshape/URDF joint names are generic:
#   revolute_1, revolute_1_1, revolute_2, etc.
#
# We need to visually identify which joint controls which leg.
#
# When the script runs:
# - Isaac Sim opens
# - the robot spawns
# - one joint wiggles for a few seconds
# - terminal prints the active joint name
# - then it moves to the next joint
#
# You watch the robot and write down what each joint does.
# ------------------------------------------------------------

import argparse

from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 1. Launch Isaac Sim first
# ------------------------------------------------------------

parser = argparse.ArgumentParser(description="Map Geronimo joints by moving one joint at a time.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ------------------------------------------------------------
# 2. Import Isaac Lab stuff after launching the app
# ------------------------------------------------------------

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass


# ------------------------------------------------------------
# 3. Your robot USD
# ------------------------------------------------------------

GERONIMO_USD = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision.usd"


# ------------------------------------------------------------
# 4. The 18 revolute joints from your URDF
# ------------------------------------------------------------

GERONIMO_MOTOR_JOINTS = [
    "revolute_1",
    "revolute_1_1",
    "revolute_1_2",
    "revolute_1_3",
    "revolute_1_4",
    "revolute_1_5",

    "revolute_2",
    "revolute_2_1",
    "revolute_2_2",
    "revolute_2_3",
    "revolute_2_4",
    "revolute_2_5",

    "revolute_3",
    "revolute_3_1",
    "revolute_3_2",
    "revolute_3_3",
    "revolute_3_4",
    "revolute_3_5",
]


# ------------------------------------------------------------
# 5. Joint mapping settings
# ------------------------------------------------------------

# How long each joint gets tested before switching to the next one.
SECONDS_PER_JOINT = 5.0

# How far each joint wiggles.
# 0.35 radians is about 20 degrees.
AMPLITUDE = 0.35

# Wiggle speed.
FREQUENCY = 1.5


# ------------------------------------------------------------
# 6. Robot config
# ------------------------------------------------------------

GERONIMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=GERONIMO_USD,

        # Disable gravity for joint mapping.
        # This keeps the robot floating so it is easier to see which leg moves.
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
        # Spawn above the ground.
        pos=(0.0, 0.0, 1.0),
    ),

    actuators={
        "leg_motors": ImplicitActuatorCfg(
            # Only the 18 real revolute joints get motors.
            joint_names_expr=GERONIMO_MOTOR_JOINTS,

            effort_limit_sim=10.0,
            velocity_limit_sim=20.0,
            stiffness=60.0,
            damping=6.0,
        )
    },
)


# ------------------------------------------------------------
# 7. Scene config
# ------------------------------------------------------------

@configclass
class JointMapperSceneCfg(InteractiveSceneCfg):
    """Simple scene with ground, light, and Geronimo."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    robot = GERONIMO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ------------------------------------------------------------
# 8. Helper function to get joint names
# ------------------------------------------------------------

def get_joint_names(robot):
    """Get joint names from the Articulation object."""

    # Most Isaac Lab versions have robot.joint_names.
    if hasattr(robot, "joint_names"):
        return list(robot.joint_names)

    # Some versions expose joint names in robot.data.
    if hasattr(robot.data, "joint_names"):
        return list(robot.data.joint_names)

    raise RuntimeError("Could not find joint names on robot or robot.data.")


# ------------------------------------------------------------
# 9. Main simulation loop
# ------------------------------------------------------------

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the joint mapping test."""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # Let Isaac initialize robot data.
    scene.update(sim_dt)

    # Get actual joint names from the loaded USD/articulation.
    actual_joint_names = get_joint_names(robot)

    print("\n[INFO] Actual joints in loaded robot:")
    for i, name in enumerate(actual_joint_names):
        print(f"  index {i:02d}: {name}")

    # Convert our desired joint names into actual joint indices.
    joint_indices = []

    for joint_name in GERONIMO_MOTOR_JOINTS:
        if joint_name not in actual_joint_names:
            print(f"[WARNING] Joint not found in USD/articulation: {joint_name}")
            continue

        joint_indices.append(actual_joint_names.index(joint_name))

    if len(joint_indices) == 0:
        raise RuntimeError("No motor joints were found. Check joint names.")

    print("\n[INFO] Starting joint mapping.")
    print("[INFO] Watch the robot. The terminal will print the active joint.")
    print("[INFO] Write down which leg/segment moves for each joint.\n")

    sim_time = 0.0
    last_active_joint = None

    while simulation_app.is_running():
        # Figure out which joint should be active right now.
        active_slot = int(sim_time // SECONDS_PER_JOINT) % len(joint_indices)
        active_joint_index = joint_indices[active_slot]
        active_joint_name = actual_joint_names[active_joint_index]

        # Print when switching to a new joint.
        if active_joint_name != last_active_joint:
            print("\n--------------------------------------------------")
            print(f"[ACTIVE JOINT] {active_joint_name}")
            print("Watch which leg moves.")
            print("--------------------------------------------------")
            last_active_joint = active_joint_name

        # Start from the robot's default joint positions.
        target = robot.data.default_joint_pos.clone()

        # Create a smooth sine wave.
        wave = AMPLITUDE * torch.sin(
            torch.tensor(sim_time * FREQUENCY, device=target.device)
        )

        # Move only the active joint.
        target[:, active_joint_index] = robot.data.default_joint_pos[:, active_joint_index] + wave

        # Send the target joint positions to the robot.
        robot.set_joint_position_target(target)

        # Step simulation.
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        scene.update(sim_dt)


# ------------------------------------------------------------
# 10. Main function
# ------------------------------------------------------------

def main():
    """Create simulation and run joint mapper."""

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Camera position.
    sim.set_camera_view(
        [2.5, -3.0, 2.0],
        [0.0, 0.0, 0.5],
    )

    scene_cfg = JointMapperSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Joint mapper loaded.")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()