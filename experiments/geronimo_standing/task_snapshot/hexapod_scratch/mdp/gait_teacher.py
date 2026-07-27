from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

FRONT_RIGHT_JOINTS = [
    "revolute_1",
    "revolute_2",
    "revolute_3",
]

MIDDLE_RIGHT_JOINTS = [
    "revolute_1_1",
    "revolute_2_1",
    "revolute_3_1",
]

BACK_RIGHT_JOINTS = [
    "revolute_1_2",
    "revolute_2_2",
    "revolute_3_2",
]

FRONT_LEFT_JOINTS = [
    "revolute_1_4",
    "revolute_2_4",
    "revolute_3_4",
]

MIDDLE_LEFT_JOINTS = [
    "revolute_1_3",
    "revolute_2_3",
    "revolute_3_3",
]

BACK_LEFT_JOINTS = [
    "revolute_1_5",
    "revolute_2_2",
    "revolute_3_2",
]

GERONIMO_MOTOR_JOINTS = (
    FRONT_RIGHT_JOINTS
    + MIDDLE_RIGHT_JOINTS
    + BACK_RIGHT_JOINTS
    + FRONT_LEFT_JOINTS
    + MIDDLE_LEFT_JOINTS
    + BACK_LEFT_JOINTS
)

MIDDLE_COXA_RANGE = 0.45
CORNER_COXA_RANGE = 0.90

FEMUR_LIFT = 0.20
TIBIA_BEND = 0.20

SWING_FRACTION = 0.35
STEP_SIGN = 1.0

GAIT_FREQUENCY = 2.0

LEG_SPECS = (
    # Front right
    (0, 0, 0.0, CORNER_COXA_RANGE, 1.0),
    # Back right
    (3, 1, -MIDDLE_COXA_RANGE, MIDDLE_COXA_RANGE, 1.0),
    # Back right
    (6, 0, -CORNER_COXA_RANGE, 0.0, 1.0),
    # Front left
    (9, 1, 0.0, CORNER_COXA_RANGE, -1.0),
    # Middle left
    (12, 0, -MIDDLE_COXA_RANGE, MIDDLE_COXA_RANGE, -1.0),
    # Back left
    (15, 1, -CORNER_COXA_RANGE, 0.0, -1.0),
)

def smoothstep(value: torch.Tensor) -> torch.Tensor:
    return value * value * (3.0 - 2.0 * value)

def get_gait_phase(
        env: ManagerBasedRLEnv,
        frequency: float = GAIT_FREQUENCY,
) -> torch.Tensor:
    
    elapsed_time = env.episode_length_buf.to(torch.float32) * float(env.step_dt)
    return torch.remained(elapsed_time * frequency, 1.0)

def gait_phase_observation(
        env: ManagerBasedRLEnv,
        frequency: float = GAIT_FREQUENCY,
) -> torch.Tensor:
    
    phase = get_gait_phase(env, frequency)
    angle = 2.0 * torch.pi * phase

    return torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
        ),
        dim=-1,
    )

def phase_offsets(
        phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    phase = torch.remainder(phase, 1.0)
    is_swing = phase < SWING_FRACTION

    swing_progress = torch.clamp(
        phase / SWING_FRACTION,
        min=0.0,
        max=1.0,
    )

    swing_progress = smoothstep(swing_progress)

    stance_progress = torch.clamp(
        (phase - SWING_FRACTION) / (1.0 - SWING_FRACTION),
        min=0.0,
        max=1.0,
    )
    stance_progress = smoothstep(stance_progress)

    coxa_progress = torch.where(
        is_swing,
        swing_progress,
        1.0 - stance_progress,
    )

    swing_lift = torch.sin(torch.pi * swing_progress)
    swing_lift = torch.where(
        is_swing,
        swing_lift,
        torch.xeros_like(swing_lift),
    )

    femur_offset = FEMUR_LIFT * swing_lift
    tibia_offset = -TIBIA_BEND * swing_lift

    return coxa_progress, femur_offset, tibia_offset

def reference_joint_offsets(
        env: ManagerBasedRLEnv,
        frequency: float = GAIT_FREQUENCY,
) -> torch.Tensor:
    
    robot = env.scene["robot"]

    offsets = torch.zeros(
        (env.num_envs, len(GERONIMO_MOTOR_JOINTS)),
        device=robot.data.joint_pos.device,
        dtype=robot.data.joint_pos.dtype,
    )

    phase_a = get_gait_phase(env, frequency)
    phase_b = torch.remainder(phase_a + 0.5, 1.0)

    coxa_a, femur_a, tibia_a = phase_offsets(phase_a)
    coxa_b, femur_b, tibia_b = phase_offsets(phase_b)

    for base_index, tripod_id, coxa_back, coxa_front, direction in LEG_SPECS:
        if tripod_id == 0:
            coxa_progress = coxa_a
            femur_offset = femur_a
            tibia_offset = tibia_a
        else:
            coxa_progress = coxa_b
            femur_offset = femur_b
            tibia_offset = tibia_b

        raw_coxa = coxa_back + (coxa_front - coxa_back) * coxa_progress

        offsets[:, base_index] = STEP_SIGN * direction * raw_coxa
        offsets[:, base_index + 1] = femur_offset
        offsets[:, base_index + 2] = tibia_offset

    return offsets