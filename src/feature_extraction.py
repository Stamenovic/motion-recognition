"""Feature extraction from filtered Vicon segment trajectories."""
from dataclasses import dataclass

import numpy as np

from .data_types import TrialRecord


LEFT_SEGMENT = "Left:left"
RIGHT_SEGMENT = "Right:right"
TRUNK_SEGMENT = "Trup:trup"


@dataclass
class TrialFeatures:
    """Time-series features extracted from one trial."""
    trial_name: str
    label: str
    time: np.ndarray
    signals: dict[str, np.ndarray]


def vector_norm(signal: np.ndarray) -> np.ndarray:
    """Return row-wise Euclidean norm for a 2D vector signal."""
    return np.linalg.norm(signal, axis=1)


def distance_between(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return Euclidean distance between two 3D trajectories through time."""
    return vector_norm(first - second)


def derivative(signal: np.ndarray, dt: float) -> np.ndarray:
    """Estimate time derivative while preserving the original signal length."""
    return np.gradient(signal, dt, axis=0)


def extract_trial_features(trial: TrialRecord) -> TrialFeatures:
    """Extract distance, speed and acceleration intensity signals from a trial."""
    missing_segments = [
        name
        for name in (LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT)
        if name not in trial.segments
    ]
    if missing_segments:
        raise KeyError(f"Trial {trial.trial_name} is missing segments: {missing_segments}")

    left = trial.segments[LEFT_SEGMENT].translation
    right = trial.segments[RIGHT_SEGMENT].translation
    trunk = trial.segments[TRUNK_SEGMENT].translation

    left_velocity = derivative(left, trial.metadata.dt)
    right_velocity = derivative(right, trial.metadata.dt)
    left_acceleration = derivative(left_velocity, trial.metadata.dt)
    right_acceleration = derivative(right_velocity, trial.metadata.dt)

    time = np.arange(trial.metadata.num_frames) * trial.metadata.dt
    return TrialFeatures(
        trial_name=trial.trial_name,
        label=trial.label,
        time=time,
        signals={
            "distance_left_right": distance_between(left, right),
            "distance_left_trunk": distance_between(left, trunk),
            "distance_right_trunk": distance_between(right, trunk),
            "speed_left": vector_norm(left_velocity),
            "speed_right": vector_norm(right_velocity),
            "acceleration_left": vector_norm(left_acceleration),
            "acceleration_right": vector_norm(right_acceleration),
        },
    )
