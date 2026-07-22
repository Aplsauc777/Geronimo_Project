# ------------------------------------------------------------
# rewards.py
#
# Purpose:
# These functions define what behavior we want the hexapod to learn.
#
# The policy does not understand "walk nicely" by default.
# It only sees numbers.
#
# Reward functions convert behavior into numbers:
#
#   good behavior -> bigger reward
#   bad behavior  -> smaller reward or penalty
#
# Our first walking objective:
# - move in the +X direction
# - match a target speed
# - stay upright
# - keep the body off the ground
# - avoid twitchy movement
# ------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_forward_speed(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.25,
    direction: tuple[float, float] = (1.0, 0.0),
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for walking in a chosen direction at a target speed.

    This is the main walking reward.

    target_speed:
        Desired speed in meters per second.

    direction:
        Desired world direction in XY.
        (1.0, 0.0) means world +X.
        (0.0, 1.0) means world +Y.

    std:
        Controls how strict the reward is.
        Smaller std = stricter speed matching.
    """

    # Get the robot object from the scene.
    robot: Articulation = env.scene[asset_cfg.name]

    # Root linear velocity in world coordinates.
    # Shape: [num_envs, 3]
    # Columns are x velocity, y velocity, z velocity.
    lin_vel_w = robot.data.root_lin_vel_w

    # Keep only horizontal velocity.
    lin_vel_xy = lin_vel_w[:, :2]

    # Convert desired direction into a torch tensor.
    desired_dir = torch.tensor(
        direction,
        device=lin_vel_xy.device,
        dtype=lin_vel_xy.dtype,
    )

    # Normalize direction so length becomes 1.
    desired_dir = desired_dir / (torch.norm(desired_dir) + 1e-6)

    # Project robot velocity onto desired direction.
    # For direction (1, 0), this is basically x velocity.
    forward_speed = torch.sum(lin_vel_xy * desired_dir, dim=1)

    # Compute sideways velocity.
    # If robot is supposed to move +X but slides in Y, this catches it.
    sideways_velocity = lin_vel_xy - forward_speed.unsqueeze(1) * desired_dir

    # Error from target speed.
    speed_error = torch.square(forward_speed - target_speed)

    # Error from sliding sideways.
    sideways_error = torch.sum(torch.square(sideways_velocity), dim=1)

    # Combine errors.
    total_error = speed_error + 0.5 * sideways_error

    # Convert error into reward.
    #
    # If error is 0, reward is close to 1.
    # If error is large, reward goes toward 0.
    return torch.exp(-total_error / (std * std + 1e-6))


def upright_reward(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for keeping its body upright."""

    robot: Articulation = env.scene[asset_cfg.name]

    # projected_gravity_b tells us the gravity direction in the robot body frame.
    #
    # If the robot is upright, the x and y components should be near 0.
    # If the robot tilts, x/y gravity components get larger.
    gravity_xy = robot.data.projected_gravity_b[:, :2]

    # Tilt error grows as the body leans.
    tilt_error = torch.sum(torch.square(gravity_xy), dim=1)

    # Turn tilt error into reward.
    return torch.exp(-tilt_error / (std * std + 1e-6))


def body_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.30,
    min_height: float = 0.18,
    std: float = 0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the robot for keeping its main body above the ground.

    This is the "do not touch the ground" part.

    Important:
    Feet can touch the ground.
    The body/chassis/root should not scrape the ground.
    """

    robot: Articulation = env.scene[asset_cfg.name]

    # Root position in world coordinates.
    # z is height above the ground.
    root_height = robot.data.root_pos_w[:, 2]

    # Reward being near the target body height.
    height_error = torch.square(root_height - target_height)

    # Extra error if the body drops below min_height.
    too_low_error = torch.square(torch.clamp(min_height - root_height, min=0.0))

    # Make "too low" much more important than just being slightly off target.
    total_error = height_error + 5.0 * too_low_error

    return torch.exp(-total_error / (std * std + 1e-6))


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize sudden changes in action.

    This encourages smoother walking.

    The policy sends an action each step.
    If the new action is very different from the last action, this gets large.
    Since this term will use a negative weight, large value = bad.
    """

    # Current action from the policy.
    action = env.action_manager.action

    # Previous action from the last step.
    prev_action = env.action_manager.prev_action

    # Sum squared action changes.
    return torch.sum(torch.square(action - prev_action), dim=1)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize very fast joint movement.

    This helps reduce violent twitching.
    """

    robot: Articulation = env.scene[asset_cfg.name]

    return torch.sum(torch.square(robot.data.joint_vel), dim=1)


def yaw_spin_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize spinning around the vertical axis.

    If the robot is supposed to walk straight, it should not constantly rotate.
    """

    robot: Articulation = env.scene[asset_cfg.name]

    # root_ang_vel_b[:, 2] is yaw angular velocity.
    yaw_rate = robot.data.root_ang_vel_b[:, 2]

    return torch.square(yaw_rate)

def forward_progress_reward(
        env: ManagerBasedRLEnv,
        target_speed: float = 0.35,
        direction: tuple[float, float] = (1.0, 0.0),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    lin_vel_xy = robot.data.root_lin_vel_w[:, :2]

    desired_dir = torch.tensor(
        direction,
        device=lin_vel_xy.device,
        dtype=lin_vel_xy.dtype,
    )

    desired_dir = desired_dir / (torch.norm(desired_dir) + 1e-6)

    forward_speed = torch.sum(lin_vel_xy * desired_dir, dim=1)

    clipped_speed = torch.clamp(forward_speed, min=0.0, max=target_speed)

    return clipped_speed / (target_speed + 1e-6)

def standing_still_penalty(
            env: ManagerBasedRLEnv,
            min_speed: float = 0.08,
            direction: tuple[float, float] = (1.0, 0.0),
            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        robot: Articulation = env.scene[asset_cfg.name]

        lin_vel_xy = robot.data.root_lin_vel_w[:, :2]

        desired_dir = torch.tensor(
            direction,
            device=lin_vel_xy.device,
            dtype=lin_vel_xy.dtype,
        )

        desired_dir = desired_dir / (torch.norm(desired_dir) + 1e-6)

        forward_speed = torch.sum(lin_vel_xy * desired_dir, dim=1)

        return torch.square(torch.clamp(min_speed - forward_speed, min=0.0))