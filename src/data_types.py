"""Shared data structures for Vicon trial loading and processing."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrialMetadata:
    fps: int
    dt: float
    num_frames: int
    frame_start: int
    frame_end: int


@dataclass
class SegmentTrajectory:
    """Rigid-body segment trajectory from the Vicon skeleton model."""
    name: str
    translation: np.ndarray  # shape: (num_frames, 3), mm (tx, ty, tz)
    rotation: np.ndarray  # shape: (num_frames, 3), deg (rx, ry, rz)


@dataclass
class TrialRecord:
    """Loaded movement trial with path-derived context and segment trajectories."""
    path: Path
    patient: str
    session: str
    trial_name: str
    label: str
    metadata: TrialMetadata
    segments: dict[str, SegmentTrajectory]
