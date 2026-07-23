# ------------------------------------------------------------
# terminations.py
#
# Purpose:
# These functions decide when an episode should end.
#
# For the hexapod:
# - End if the body/root gets too close to the ground.
# - End if the robot tilts/flips too much.
# ------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def body_too_low(
    env: ManagerBasedRLEnv,
    min_height: float = 0.12,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode if the robot body/root is too close to the ground."""

    # Get the robot from the scene.
    robot: Articulation = env.scene[asset_cfg.name]

    # Get the z position of the robot root/body.
    root_height = robot.data.root_pos_w[:, 2]

    # Return True for environments where the body is too low.
    return root_height < min_height


def bad_orientation(
    env: ManagerBasedRLEnv,
    max_tilt: float = 0.8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End the episode if the robot tilts too much."""

    # Get the robot from the scene.
    robot: Articulation = env.scene[asset_cfg.name]

    # projected_gravity_b tells us gravity direction in the robot body frame.
    # If the robot is upright, x and y should be near 0.
    gravity_xy = robot.data.projected_gravity_b[:, :2]

    # Tilt amount gets larger as the body leans/flips.
    tilt_amount = torch.linalg.norm(gravity_xy, dim=1)

    # Return True for environments where tilt is too large.
    return tilt_amount > max_tilt