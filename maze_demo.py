"""
Geronimo - MAZE BUILDER + RADAR DEMO

Builds a maze from ASCII art as a SINGLE UsdGeom.Mesh prim, which is what
Isaac Lab's RayCaster needs (it refuses more than one mesh path).

Edit MAZE below - '#' is wall, '.' is floor, 'S' is the robot start,
'P' is where a person stands. Then:

    python maze_demo.py                # radar only
    python maze_demo.py --camera       # radar + camera person-detection

Close the window to exit.
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--camera", action="store_true", help="Also run camera detection.")
parser.add_argument("--cell", type=float, default=1.2, help="Cell size in metres.")
parser.add_argument("--wall_h", type=float, default=1.2, help="Wall height in metres.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.camera:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is up."""

import math
import os
import sys

import cv2
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, Vt

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

sys.path.insert(0, r"C:\D drive\Local Disk D\Kush 2\Project Geronimo\Vision Task\Geronimo_Project")
from surround import analyze_surroundings, choose_heading, draw_radar

GERONIMO_USD = r"C:\D drive\Local Disk D\Kush 2\Project Geronimo\Geronimo_Project-geronimo-standing\Geronimo_Project-geronimo-standing\simulation\assets\geronimo_v3_collision_updated_usd.usd"
PERSON_USD = (f"{ISAAC_NUCLEUS_DIR}/People/Characters/"
              f"original_male_adult_construction_03/male_adult_construction_03.usd")
OUT_DIR = r"C:\IsaacLab"

# ---------------------------------------------------------------------------
# THE MAZE. '#'=wall  '.'=floor  'S'=robot start  'P'=person
# Row 0 is the far side (+X); the robot faces +X, i.e. "up" this diagram.
# ---------------------------------------------------------------------------
MAZE = [
    "#########",
    "#...#..P#",
    "#.#.#.###",
    "#.#...#.#",
    "#.#####.#",
    "#...S...#",
    "#########",
]


def _box_tris(cx, cy, half, h):
    """8 corners + 12 triangles for one wall block.
    Returns (points, triangle_index_triples). Triangles, not quads, because
    the warp raycaster wants triangles."""
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    pts = [
        (x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0),   # bottom 0-3
        (x0, y0, h), (x1, y0, h), (x1, y1, h), (x0, y1, h),           # top    4-7
    ]
    quads = [
        (0, 3, 2, 1),   # bottom
        (4, 5, 6, 7),   # top
        (0, 1, 5, 4),   # -Y side
        (1, 2, 6, 5),   # +X side
        (2, 3, 7, 6),   # +Y side
        (3, 0, 4, 7),   # -X side
    ]
    tris = []
    for a, b, c, d in quads:
        tris.append((a, b, c))      # split each quad into two triangles
        tris.append((a, c, d))
    return pts, tris


def build_maze(layout, cell=1.2, wall_h=1.2, prim_path="/World/Maze"):
    """Create ONE Mesh prim containing every wall block.

    Returns (prim_path, cell_to_world) where cell_to_world(row, col) -> (x, y).
    """
    rows = len(layout)
    cols = max(len(r) for r in layout)

    # Centre the maze on the origin, and make row 0 the +X (forward) end.
    def cell_to_world(row, col):
        x = (rows / 2.0 - row - 0.5) * cell
        y = (cols / 2.0 - col - 0.5) * cell
        return x, y

    all_pts, all_tris = [], []
    for r, line in enumerate(layout):
        for c, ch in enumerate(line):
            if ch != "#":
                continue
            cx, cy = cell_to_world(r, c)
            pts, tris = _box_tris(cx, cy, cell / 2.0, wall_h)
            base = len(all_pts)                 # offset indices into the big array
            all_pts.extend(pts)
            all_tris.extend([(a + base, b + base, c2 + base) for a, b, c2 in tris])

    # --- write it as a single USD Mesh -------------------------------------
    stage = omni.usd.get_context().get_stage()
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in all_pts]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(all_tris)))
    mesh.CreateFaceVertexIndicesAttr(
        Vt.IntArray([i for tri in all_tris for i in tri]))

    # Extent helps the renderer cull correctly.
    arr = np.array(all_pts, dtype=float)
    mesh.CreateExtentAttr(Vt.Vec3fArray([
        Gf.Vec3f(*arr.min(axis=0).tolist()), Gf.Vec3f(*arr.max(axis=0).tolist())]))

    # Collision, so the robot can't walk through walls later.
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())

    print(f"[MAZE] {len(all_pts)} points, {len(all_tris)} triangles -> {prim_path}",
          flush=True)
    return prim_path, cell_to_world


def find_cells(layout, ch):
    """Every (row, col) holding character `ch`."""
    return [(r, c) for r, line in enumerate(layout)
            for c, x in enumerate(line) if x == ch]


def main():
    cell, wall_h = args_cli.cell, args_cli.wall_h
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))

    # --- ground + light -----------------------------------------------------
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.9, 0.9, 0.9))
    light.func("/World/Light", light)

    # --- the maze (ONE mesh - this is the whole point) ----------------------
    maze_path, cell_to_world = build_maze(MAZE, cell=cell, wall_h=wall_h)

    # --- robot at 'S' -------------------------------------------------------
    starts = find_cells(MAZE, "S")
    sx, sy = cell_to_world(*starts[0]) if starts else (0.0, 0.0)
    robot_cfg = sim_utils.UsdFileCfg(usd_path=GERONIMO_USD)
    robot_cfg.func("/World/Robot", robot_cfg, translation=(sx, sy, 0.35))
    print(f"[MAZE] robot at cell {starts[0] if starts else '(0,0)'} -> ({sx:.2f}, {sy:.2f})",
          flush=True)

    # --- person at 'P' ------------------------------------------------------
    people = find_cells(MAZE, "P")
    if people:
        px, py = cell_to_world(*people[0])
        person = sim_utils.UsdFileCfg(usd_path=PERSON_USD)
        person.func("/World/Person", person, translation=(px, py, 0.0),
                    orientation=(0.7071, 0.0, 0.0, -0.7071))
        dist = math.hypot(px - sx, py - sy)
        print(f"[MAZE] person at ({px:.2f}, {py:.2f}) - {dist:.2f} m from robot",
              flush=True)
    
    # --- 360 scan against the maze mesh -------------------------------------
    scan = RayCaster(RayCasterCfg(
        prim_path="/World/Robot/root",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.15)),
        attach_yaw_only=True,
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1, vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0),
        max_distance=12.0,
        debug_vis=True,
        mesh_prim_paths=[maze_path],       # ONE mesh - the whole maze
    ))

    # --- optional camera ----------------------------------------------------
    camera = None
    FRONT_OFFSET_DEG = -90.0
    _h = math.radians(FRONT_OFFSET_DEG) / 2
    FRONT_QUAT = (math.cos(_h), 0.0, 0.0, math.sin(_h))
    if args_cli.camera:
        camera = Camera(CameraCfg(
            prim_path="/World/Robot/root/head_cam",
            offset=CameraCfg.OffsetCfg(pos=(0.10, 0.0, 0.12),
                                       rot=FRONT_QUAT, convention="world"),
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.1, 30.0)),
            width=640, height=480,
        ))

    sim.reset()
    print("[MAZE] sim.reset() OK", flush=True)

    # YOLO AFTER sim.reset() - loading it earlier breaks the render pipeline.
    model = None
    if args_cli.camera:
        from ultralytics import settings as ul_settings
        ul_settings.update({"sync": False})        # no telemetry thread
        from ultralytics import YOLO
        model = YOLO("yolo11n-pose.pt")
        print("[MAZE] YOLO loaded", flush=True)

    step, saved = 0, False
    while simulation_app.is_running():
        sim.step()
        scan.update(sim.get_physics_dt())
        if camera is not None:
            camera.update(sim.get_physics_dt())
        step += 1

        # --- rays -> robot-relative bearings ---------------------------------
        hits = scan.data.ray_hits_w[0].cpu().numpy()
        origin = scan.data.pos_w[0].cpu().numpy()
        quat = scan.data.quat_w[0].cpu().numpy()
        yaw = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
        rel = hits[:, :2] - origin[:2]
        distances = np.linalg.norm(rel, axis=1)
        angles = np.arctan2(rel[:, 1], rel[:, 0]) - yaw - math.radians(FRONT_OFFSET_DEG)
        state = analyze_surroundings(angles, distances, max_range=6.0)
        fwd, yawcmd = choose_heading(state)

        if step == 60 and not saved:
            radar = draw_radar(state, size=420)
            cv2.putText(radar, f"fwd={fwd:.2f} yaw={yawcmd:+.2f}", (8, 410),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(OUT_DIR, "maze_radar.png"), radar)
            print(f"[MAZE] {state.summary()}", flush=True)
            print(f"[MAZE] hits in range: {int((distances < 6.0).sum())} / {len(distances)}",
                  flush=True)
            print(f"[MAZE] decision: fwd={fwd:.2f} yaw={yawcmd:+.2f}", flush=True)

            if camera is not None and model is not None:
                rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
                frame = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                res = model(frame, classes=[0], conf=0.30, verbose=False)[0]
                n = 0 if res.boxes is None else len(res.boxes)
                for b in (res.boxes.xyxy.cpu().numpy() if n else []):
                    cv2.rectangle(frame, (int(b[0]), int(b[1])),
                                  (int(b[2]), int(b[3])), (0, 255, 0), 2)
                cv2.imwrite(os.path.join(OUT_DIR, "maze_camera.png"), frame)
                print(f"[MAZE] camera: {n} person(s) detected", flush=True)

            print("[MAZE] saved. window stays open - close it to exit", flush=True)
            saved = True

    print("[MAZE] done", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()