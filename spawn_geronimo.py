# ------------------------------------------------------------
# spawn_geronimo.py
#
# Purpose:
# This script opens Isaac Sim through Isaac Lab, loads your
# Geronimo hexapod USD file, spawns it into a simple world,
# prints the joint names, and wiggles the joints so you can
# confirm the robot imported correctly.
# ------------------------------------------------------------


# argparse lets us read command-line arguments like:
# --num_envs 1
# --headless
# --device cuda:0
import argparse


# AppLauncher is the Isaac Lab tool that starts Isaac Sim.
# You always need this before importing most Isaac Lab simulation modules.
from isaaclab.app import AppLauncher


# ------------------------------------------------------------
# 1. Parse command-line arguments
# ------------------------------------------------------------

# Create an argument parser for this script.
parser = argparse.ArgumentParser(description="Spawn Geronimo hexapod in Isaac Lab.")

# Add our own custom argument.
# num_envs means how many copies of the robot/world to spawn.
# For visual testing, keep this at 1.
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of Geronimo environments to spawn."
)

# Add Isaac Lab's built-in app arguments.
# This adds options like --headless, --device, --livestream, etc.
AppLauncher.add_app_launcher_args(parser)

# Actually read the command-line arguments.
args_cli = parser.parse_args()


# ------------------------------------------------------------
# 2. Launch Isaac Sim
# ------------------------------------------------------------

# This creates the Isaac Sim application.
# Nothing simulation-related should be imported before this point,
# because Isaac Sim needs to start first.
app_launcher = AppLauncher(args_cli)

# This is the actual running Isaac Sim app.
simulation_app = app_launcher.app


# ------------------------------------------------------------
# 3. Import Isaac Lab and PyTorch tools
# ------------------------------------------------------------

# torch is used for tensors.
# Isaac Lab stores robot states, actions, joint positions, etc. as tensors.
import torch

# sim_utils contains simulation config objects:
# ground planes, lights, USD spawning, physics settings, etc.
import isaaclab.sim as sim_utils

# ImplicitActuatorCfg defines simple joint motors/drives.
# We use this so Isaac Lab can command the robot's joints.
from isaaclab.actuators import ImplicitActuatorCfg

# ArticulationCfg is used for robots with joints.
# AssetBaseCfg is used for simple scene assets like lights and ground.
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

# InteractiveScene and InteractiveSceneCfg help organize the world.
# They let you define a scene with ground, lights, robots, etc.
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


# ------------------------------------------------------------
# 4. Path to your imported hexapod USD
# ------------------------------------------------------------

# This is the USD file you created from the URDF converter.
# The r before the string means "raw string", so Windows backslashes work.
GERONIMO_USD = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision.usd"
# ------------------------------------------------------------
# 5. Define Geronimo as an Isaac Lab articulation
# ------------------------------------------------------------

# ArticulationCfg tells Isaac Lab:
# "This is a robot with movable joints."
GERONIMO_CFG = ArticulationCfg(

    # spawn controls how the robot is loaded into the simulation.
    spawn=sim_utils.UsdFileCfg(

        # This is the actual robot file.
        usd_path=GERONIMO_USD,

        # Rigid body physics settings.
        # These affect how the robot's physical links behave.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(

            # False means gravity affects the robot.
            disable_gravity=False,

            # Helps prevent extreme physics correction speeds
            # when objects start slightly inside each other.
            max_depenetration_velocity=5.0,
        ),

        # Articulation physics settings.
        # These apply to the whole jointed robot.
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(

            # False means the robot's own links will not collide with each other.
            # This is usually safer for first import tests because CAD collisions
            # can overlap and make the robot explode.
            enabled_self_collisions=False,

            # More position iterations usually makes joints more stable.
            solver_position_iteration_count=8,

            # Velocity iterations can usually start low.
            solver_velocity_iteration_count=1,
        ),
    ),

    # Initial state of the robot when it spawns.
    init_state=ArticulationCfg.InitialStateCfg(

        # Spawn position in meters: x, y, z.
        # z = 0.35 means the robot starts slightly above the ground.
        pos=(0.0, 0.0, 1.0),
    ),

    # Actuators tell Isaac Lab how to command the joints.
    actuators={

        # This name is arbitrary. We are making one actuator group
        # that controls all joints.
        "all_joints": ImplicitActuatorCfg(

            # This regular expression means "match every joint name."
            # Later, you can replace this with only the leg servo joints.
            joint_names_expr=[".*"],

            # Maximum motor effort/torque.
            # If the robot is too weak, increase this.
            # If it explodes or shakes badly, lower it.
            effort_limit_sim=10.0,

            # Maximum joint speed.
            velocity_limit_sim=20.0,

            # Stiffness controls how strongly joints try to reach the target position.
            # Higher = stronger/snappier.
            stiffness=20.0,

            # Damping resists motion and helps reduce shaking.
            damping=2.0,
        )
    },
)


# ------------------------------------------------------------
# 6. Define the whole scene
# ------------------------------------------------------------

# This class defines what exists in the world.
# It is like saying:
# "My scene has a ground plane, a light, and my robot."
class GeronimoSceneCfg(InteractiveSceneCfg):

    # Add a default ground plane at /World/defaultGroundPlane.
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Add a dome light so the robot is visible.
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    # Add Geronimo to each environment.
    # {ENV_REGEX_NS} lets Isaac Lab spawn multiple copies cleanly
    # when num_envs is more than 1.
    robot = GERONIMO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ------------------------------------------------------------
# 7. Main simulation loop
# ------------------------------------------------------------

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation after the robot and scene are created."""

    # Get the robot object from the scene.
    robot = scene["robot"]

    # Get the physics timestep.
    # Example: if sim_dt = 0.005, physics runs at 200 Hz.
    sim_dt = sim.get_physics_dt()

    # count tracks how many simulation steps have happened.
    count = 0

    # sim_time tracks time in seconds.
    sim_time = 0.0

    # This makes sure we only print joint names once.
    printed_joints = False

    # This loop keeps running while the Isaac Sim window is open.
    while simulation_app.is_running():

        # ----------------------------------------------------
        # Reset robot every 500 simulation steps
        # ----------------------------------------------------
        if count % 500 == 0:

            # Get the default root state.
            # Root state includes position, orientation, linear velocity,
            # and angular velocity of the robot base.
            root_state = robot.data.default_root_state.clone()

            # Add environment origins so each copy spawns in the right place.
            root_state[:, :3] += scene.env_origins

            # Write the base pose to the simulator.
            robot.write_root_pose_to_sim(root_state[:, :7])

            # Write the base velocity to the simulator.
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # Get default joint positions and velocities.
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()

            # Reset all joints to their default positions/velocities.
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # Tell the scene that everything has been reset.
            scene.reset()

            print("[INFO] Reset Geronimo.")

        # ----------------------------------------------------
        # Print joint names once
        # ----------------------------------------------------
        if not printed_joints:

            print("[INFO] Joint names:")

            # Different Isaac Lab versions may store joint names slightly differently.
            # This tries the most common place first.
            if hasattr(robot, "joint_names"):
                print(robot.joint_names)
            elif hasattr(robot.data, "joint_names"):
                print(robot.data.joint_names)
            else:
                print("[WARN] Could not find joint_names attribute.")

            printed_joints = True

        # ----------------------------------------------------
        # Make a simple test joint command
        # ----------------------------------------------------

        # Start from the default joint positions.
        target = robot.data.default_joint_pos.clone()

        # Create a sine wave number.
        # This changes smoothly over time between about -1 and +1.
        wave = torch.sin(torch.tensor(sim_time * 2.0, device=target.device))

        # Move every joint by a small amount using that sine wave.
        # 0.25 means about 0.25 radians, which is about 14 degrees.
        target[:] = target + 0.25 * wave

        # Send those target joint positions to the robot.
        # This only works if the joints have position actuators/drives.
        robot.set_joint_position_target(target)

        # ----------------------------------------------------
        # Step the simulation
        # ----------------------------------------------------

        # Write all pending robot commands into the simulator.
        scene.write_data_to_sim()

        # Advance physics by one timestep.
        sim.step()

        # Update our time and step counter.
        sim_time += sim_dt
        count += 1

        # Update Isaac Lab's internal buffers after the physics step.
        scene.update(sim_dt)


# ------------------------------------------------------------
# 8. Main function
# ------------------------------------------------------------

def main():
    """Creates the simulation, creates the scene, and starts the loop."""

    # Create basic simulation settings.
    # args_cli.device usually becomes something like "cuda:0" or "cpu".
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)

    # Create the simulation context.
    # This controls physics stepping, rendering, camera, etc.
    sim = sim_utils.SimulationContext(sim_cfg)

    # Set the camera view.
    # First list = camera position.
    # Second list = where the camera looks.
    sim.set_camera_view(
        [2.5, -3.0, 2.0],
        [0.0, 0.0, 0.25]
    )

    # Create the scene configuration.
    # num_envs controls how many copies of the scene to spawn.
    # env_spacing controls how far apart they are.
    scene_cfg = GeronimoSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
    )

    # Build the actual scene from the scene config.
    scene = InteractiveScene(scene_cfg)

    # Reset the simulation so all objects are fully initialized.
    sim.reset()

    print("[INFO] Geronimo spawn test loaded.")

    # Start running the simulation loop.
    run_simulator(sim, scene)


# ------------------------------------------------------------
# 9. Script entry point
# ------------------------------------------------------------

# This makes sure main() only runs when this file is executed directly.
if __name__ == "__main__":

    # Run the main function.
    main()

    # Close Isaac Sim cleanly when the script ends.
    simulation_app.close()