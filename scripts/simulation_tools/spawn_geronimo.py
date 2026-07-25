
import argparse


from isaaclab.app import AppLauncher




parser = argparse.ArgumentParser(description="Spawn Geronimo hexapod in Isaac Lab.")

parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of Geronimo environments to spawn."
)


AppLauncher.add_app_launcher_args(parser)


args_cli = parser.parse_args()



app_launcher = AppLauncher(args_cli)

simulation_app = app_launcher.app





import torch
import math

import isaaclab.sim as sim_utils


from isaaclab.actuators import ImplicitActuatorCfg


from isaaclab.assets import ArticulationCfg, AssetBaseCfg


from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from isaaclab.sensors import ContactSensorCfg

import json
from pathlib import Path


GERONIMO_USD = r"C:\Users\Johnt\OneDrive\Desktop\Hexapod\assets_v3\geronimo_v3_collision_heavy.usd"

PROJECT_ROOT = Path(
    r"C:\Users\Johnt\OneDrive\Desktop\GitHub_Geronimo\Geronimo_Project"
)

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

        "all_joints": ImplicitActuatorCfg(

            joint_names_expr=[".*"],

            effort_limit_sim=100.0,

            velocity_limit_sim=20.0,

            stiffness=20.0,

            damping=2.0,
            armature=0.01,
        )
    },
)



class GeronimoSceneCfg(InteractiveSceneCfg):

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )


    robot = GERONIMO_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    all_body_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
        debug_vis=False,
    )




def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation after the robot and scene are created."""

    robot = scene["robot"]

    contact_sensor = scene["all_body_contact_sensor"]


    sim_dt = sim.get_physics_dt()

    count = 0

    sim_time = 0.0

    print_interval_steps = max(1, round(1.0 / sim_dt))

    previous_touching_bodies = None

    print("\nBodies monitored by the contanct sensor:")
    for index, body_name in enumerate(contact_sensor.body_names):
        print(f"{index}: {body_name}")

    while simulation_app.is_running():

 
        scene.write_data_to_sim()

        sim.step()

        sim_time += sim_dt

        scene.update(sim_dt)

        count += 1

        if count % print_interval_steps == 0:
            body_forces = contact_sensor.data.net_forces_w[0]

            force_magnitudes = torch.linalg.vector_norm(body_forces, dim=-1)

            contact_threshold = 0.5

            touching_bodies = []
            previous_touching_bodies = tuple()
            for body_name, force in zip(contact_sensor.body_names, force_magnitudes):
                force_value = force.item()
                if force_value > contact_threshold:
                    touching_bodies.append((body_name, force_value))

                current_touching_names = tuple(body_name for body_name, _ in touching_bodies)

                if current_touching_names != previous_touching_bodies:
                    print("\n" + "=" * 65)
                    print(f"bodies in contact at t={sim_time:.2f} seconds")
                    print("=" * 65)

                    if not touching_bodies:
                        print("No robot bodies detected in contact.")
                    else:
                        for body_name, force_value in touching_bodies:
                            print(f"{body_name}: force={force_value}")

                    previous_touching_bodies = current_touching_names




def main():
    """Creates the simulation, creates the scene, and starts the loop."""


    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)


    sim = sim_utils.SimulationContext(sim_cfg)


    sim.set_camera_view(
        [2.5, -3.0, 2.0],
        [0.0, 0.0, 0.25]
    )


    scene_cfg = GeronimoSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
    )

    scene = InteractiveScene(scene_cfg)

    sim.reset()

    print("[INFO] Geronimo spawn test loaded.")

    run_simulator(sim, scene)




if __name__ == "__main__":

    main()

    simulation_app.close()