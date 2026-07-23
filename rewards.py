from __future__ import annotations

import math

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def track_forward_speed_exp(env: ManagerBasedRLEnv, target_speed: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    actual_forward_spped = robot.data.root_lin_vel_b[:, 0]
    speed_error = torch.square(actual_forward_spped - target_speed)
    return torch.exp(-speed_error / (std * std))

def upright_posture_exp(env: ManagerBasedRLEnv, std: float, target_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    tilt_error = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    return torch.exp(-tilt_error / (std * std))



def body_height_exp(env: ManagerBasedRLEnv, target_height: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    actual_height = robot.data.root_link_pos_w[:, 2]
    height_error = torch.square(actual_height - target_height)
    return torch.exp(-height_error / (std * std))

def lateral_velocity_12(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_lin_vel_b[:, 2])

def roll_pitch_rate_12(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(robot.data.root_ang_vel_b[:, :2]), dim=1)

def yaw_rate_12(env: ManagerBasedRLEnv, target_yaw_rate: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.square(robot.data.root_ang_vel_b[:, 2] - target_yaw_rate)

def tripod_reference_tracking_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:

    robot: Articulation = env.scene[asset_cfg.name]

    if not hasattr(env, "gait_reference_joint_pos"):
        raise RuntimeError("gait_reference_joint_pos is not set in the environment. Please ensure that the environment has a gait reference trajectory.")
    
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    reference = env.gait_reference_joint_pos
    error = torch.mean(torch.square(joint_pos - reference), dim=1)

    return torch.exp(-error / (std * std))

def gait_phase_observation(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    time = (env.episode_length_buf.float() * env.step_dt)
    phase = time / cycle_time * 1.0
    angle = 2.0 * math.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    