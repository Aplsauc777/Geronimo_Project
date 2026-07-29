from __future__ import annotations

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

import torch

_TRIPOD_PHASE_OFFSETS = (0.0, 0.5, 0.0, 0.5, 0.0, 0.5)

def _smooth_tripod_stance_weight(env: ManagerBasedRLEnv, cycle_time: float, duty_factor: float, transition_fraction: float = 0.05) -> torch.Tensor:
    global_phase = _global_gait_phase(env, cycle_time)
    offsets = torch.tensor(_TRIPOD_PHASE_OFFSETS, device=env.device, dtype=global_phase.dtype)
    foot_phases = torch.remainder(global_phase.unsqueeze(1) + offsets, 1.0)
    stance_center = 0.5 * duty_factor
    circular_offset = torch.remainder(foot_phases - stance_center + 0.5, 1.0) - 0.5
    distance_from_center = torch.abs(circular_offset)
    stance_half_width = 0.5 * duty_factor
    blend = (stance_half_width + transition_fraction - distance_from_center) / (2.0 * transition_fraction)

    blend = torch.clamp(blend, min=0.0, max=1.0)
    stance_weight = blend * blend * (3.0 - 2.0 * blend)
    return stance_weight

def _global_gait_phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:

    if cycle_time <= 0.0:
        raise ValueError("how did you manage to get a negative cycle time")

    elapsed_time = env.episode_length_buf.float() * env.step_dt
    return torch.remainder(elapsed_time / cycle_time, 1.0)

def gait_phase_observation(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:

    phase = _global_gait_phase(env, cycle_time)
    angle = 2.0 * torch.pi * phase
    return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

def _desired_tripod_stance(env: ManagerBasedRLEnv, cycle_time: float, duty_factor: float) -> torch.Tensor:

    if not 0.5 <= duty_factor < 1.0:
        raise ValueError("nice job passing a duty_factor not in the range of [0.5, 1.0)")

    global_phase = _global_gait_phase(env, cycle_time)
    offsets = torch.tensor(_TRIPOD_PHASE_OFFSETS, device=env.device, dtype=global_phase.dtype)
    foot_phases = torch.remainder(global_phase.unsqueeze(1) + offsets, 1.0)
    return foot_phases < duty_factor

def tripod_swing_force_reward(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, cycle_time: float, duty_factor: float = 0.55, force_std: float = 1.0, phase_transition_fraction: float = 0.08, command_threshold: float = 0.1) -> torch.Tensor:
    if force_std <= 0.0:
        raise ValueError("you do know that force_std has to be positive right?")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    stance_weight = _smooth_tripod_stance_weight(env, cycle_time=cycle_time, duty_factor=duty_factor, transition_fraction=0.05)

    

    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.norm(force_history, dim=-1).max(dim=1).values

    swing_weight = 1.0 - stance_weight
    per_foot_reward = torch.exp(-torch.square(force_magnitude / force_std))

    swing_count = swing_weight.sum(dim=1).clamp(min=1.0)
    reward = (per_foot_reward * swing_weight).sum(dim=1) / swing_count


    command = env.command_manager.get_command(command_name)
    command_magnitude = torch.linalg.vector_norm(command, dim=1)
    moving_command = command_magnitude > command_threshold
    return reward * moving_command.to(reward.dtype)

def tripod_stance_velocity_reward(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, cycle_time: float, duty_factor: float = 0.55, velocity_std: float = 0.05, contact_threshold: float = 1.0, transition_width: float = 0.2, phase_transition_fraction: float = 0.08, command_threshold: float = 0.1) -> torch.Tensor:
    if velocity_std <= 0.0:
        raise ValueError("you do know that velocity_std has to be positive right?")

    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    stance_weight = _smooth_tripod_stance_weight(env, cycle_time, duty_factor, phase_transition_fraction)

    foot_velocity_xy = robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    velocity_error = torch.sum(torch.square(foot_velocity_xy), dim=-1)
    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.vector_norm(force_history, dim=-1).max(dim=1).values

    contact_score = torch.sigmoid((force_magnitude - contact_threshold) / transition_width)
    low_velocity_score = torch.exp(-velocity_error / (velocity_std * velocity_std))

    per_foot_score = low_velocity_score * contact_score

    stance_count = stance_weight.sum(dim=1).clamp(min=1.0)
    reward = (per_foot_score * stance_weight).sum(dim=1) / stance_count

    command = env.command_manager.get_command(command_name)

    command_magnitude = torch.linalg.vector_norm(command, dim=1)
    moving_command = command_magnitude > command_threshold

    return reward * moving_command.to(reward.dtype)

def tripod_stance_contact_reward(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, cycle_time: float, duty_factor: float = 0.55, contact_threshold: float = 1.0, transition_width: float = 0.5, worst_foot_weight: float = 0.5, phase_transition_fraction: float = 0.05, command_threshold: float = 0.1) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    stance_weight = _smooth_tripod_stance_weight(env, cycle_time=cycle_time, duty_factor=duty_factor, transition_fraction=phase_transition_fraction)
    force_history = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    force_magnitude = torch.linalg.vector_norm(force_history, dim=-1).max(dim=1).values

    contact_score = torch.sigmoid((force_magnitude - contact_threshold) / transition_width)

    stance_count = stance_weight.sum(dim=1).clamp(min=1.0)

    mean_stance_score = (contact_score * stance_weight).sum(dim=1) /stance_count

    per_foot_stance_score = 1.0 - stance_weight * (1.0 - contact_score)

    worst_stance_score = torch.amin(per_foot_stance_score, dim=1)

    reward = (1.0 - worst_foot_weight) * mean_stance_score + worst_foot_weight * worst_stance_score
    command = env.command_manager.get_command(command_name)
    command_magnitude = torch.linalg.vector_norm(command, dim=1)
    moving_command = command_magnitude > command_threshold
    return reward * moving_command.to(reward.dtype)

