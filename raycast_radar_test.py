import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is up."""

import math
import os
import sys

import cv2
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext

sys.path.insert(0, r"C:\D drive\Local Disk D\Kush 2\Project Geronimo\Vision Task\Geronimo_Project")
from surround import analyze_surroundings, choose_heading, draw_radar

GERONIMO_USD = r"C:\D drive\Local Disk D\Kush 2\Project Geronimo\Geronimo_Project-geronimo-standing\Geronimo_Project-geronimo-standing\simulation\assets\geronimo_v3_collision_updated_usd.usd"
OUT_DIR = r"C:\IsaacLab"

# (name, bearing_deg, distance_m) - bearing 0 = forward, +ve = left
OBSTACLES = [
    ("box_front", 0.0, 2.0),
    ("box_left", 90.0, 1.5),
    ("box_right", -60.0, 2.5),
    ("box_back", 180.0, 3.0),
]


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))

    # --- ground + light -----------------------------------------------------
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)

    # --- the robot (so the scan attaches where it will on the real thing) ---
    robot_cfg = sim_utils.UsdFileCfg(usd_path=GERONIMO_USD)
    robot_cfg.func("/World/Robot", robot_cfg, translation=(0.0, 0.0, 0.3))
    print("[RAY] robot spawned", flush=True)

    # --- obstacle boxes at known bearings -----------------------------------
    # Robot forward is +X, left is +Y. Convert bearing/distance -> x, y.
    

    for name, bearing_deg, dist in OBSTACLES:
        a = math.radians(bearing_deg)
        x, y = dist * math.cos(a), dist * math.sin(a)
        box = sim_utils.MeshCuboidCfg(
            size=(0.4, 0.4, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.3, 0.2)),
        )
        path = f"/World/Obstacles/{name}"
        box.func(path, box, translation=(x, y, 0.5))
        print(f"[RAY] {name}: bearing {bearing_deg:+.0f} deg, {dist} m -> ({x:.2f}, {y:.2f})",
              flush=True)
        mesh_paths = ["/World/Obstacles/box_front/geometry/mesh"]
    # --- what's actually in the stage? --------------------------------------
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith("/World/Obstacles"):
            print(f"[STAGE] {p}   type={prim.GetTypeName()}", flush=True)
    # --- the 360-degree scan ------------------------------------------------
    # mesh_prim_paths is THE critical setting: rays only hit meshes listed here.
    # Miss one and that obstacle is invisible to the scan.
    scan = RayCaster(
        RayCasterCfg(
            prim_path="/World/Robot/root",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
            attach_yaw_only=True,          # ignore body pitch/roll wobble
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=5.0,        # 72 rays
            ),
            max_distance=8.0,
            debug_vis=True,                # draw rays in the viewport
            mesh_prim_paths=mesh_paths,
        )
    )
    # from isaaclab.terrains import TerrainImporter, TerrainImporterCfg, TerrainGeneratorCfg
    # import isaaclab.terrains as terrain_gen

    # terrain = TerrainImporter(TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=TerrainGeneratorCfg(
    #         size=(10.0, 10.0),
    #         num_rows=1, num_cols=1,
    #         sub_terrains={
    #             "rubble": terrain_gen.HfDiscreteObstaclesTerrainCfg(
    #                 num_obstacles=20,
    #                 obstacle_width_range=(0.3, 0.8),
    #                 obstacle_height_range=(0.4, 1.2),
    #                 platform_width=2.0,      # clear spawn area in the middle
    #             ),
    #         },
    #     ),
    #     )
    # )   
    print(f"[RAY] scan created against {len(mesh_paths)} meshes", flush=True)

    sim.reset()
    print("[RAY] sim.reset() OK", flush=True)

    for step in range(args_cli.frames):
        sim.step()
        scan.update(sim.get_physics_dt())

        hits = scan.data.ray_hits_w[0].cpu().numpy()      
        origin = scan.data.pos_w[0].cpu().numpy()         
        quat = scan.data.quat_w[0].cpu().numpy()         

        yaw = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))

        rel = hits[:, :2] - origin[:2]                    
        distances = np.linalg.norm(rel, axis=1)
        angles = np.arctan2(rel[:, 1], rel[:, 0]) - yaw   

        state = analyze_surroundings(angles, distances, max_range=6.0)
        fwd, yawcmd = choose_heading(state)

        if step == 60:
            radar = draw_radar(state, size=420)
            cv2.putText(radar, f"fwd={fwd:.2f} yaw={yawcmd:+.2f}", (8, 410),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            out = os.path.join(OUT_DIR, "raycast_radar.png")
            cv2.imwrite(out, radar)
            print(f"[RAY] saved {out}", flush=True)
            print(f"[RAY] {state.summary()}", flush=True)
            print(f"[RAY] finite hits: {np.isfinite(distances).sum()} / {len(distances)}",
                  flush=True)
            print(f"[RAY] decision: fwd={fwd:.2f} yaw={yawcmd:+.2f}", flush=True)
            break

    print("[RAY] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()