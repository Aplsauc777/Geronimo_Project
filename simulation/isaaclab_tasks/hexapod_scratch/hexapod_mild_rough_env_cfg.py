from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from . import mdp
from .hexapod_forward_env_cfg import HexapodForwardEnvCfg

GERONIMO_MILD_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.0025,
    slope_threshold=0.75,
    use_cache=False,

    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),

        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.0025, 0.015),
            noise_step=0.0025,
            border_width=0.25,
        ),

        "gentle_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.08),
            platform_width=1.5,
            border_width=0.25,
        ),

        "gentle_slope_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.08),
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)

@configclass
class MildTerrainCurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_vel
    )

@configclass
class HexapodMildRoughEnvCfg(HexapodForwardEnvCfg):

    curriculum: MildTerrainCurriculumCfg = MildTerrainCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=GERONIMO_MILD_TERRAINS_CFG,

            max_init_terrain_level=0,

            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )

        self.commands.base_velocity.ranges.lin_vel_x = (0.1, 0.22)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,  0.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (0.0, 0.0)
        self.scene.terrain.max_init_terrain_level = 0 