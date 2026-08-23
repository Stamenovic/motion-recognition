"""Utilities for turning streamed Vicon frames into TrialRecord segments."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data_types import SegmentTrajectory, TrialMetadata, TrialRecord


@dataclass
class LiveFrame:
    """One streamed frame with segment translations and optional rotations."""

    frame_number: int
    timestamp_seconds: float
    translations: dict[str, tuple[float, float, float]]
    rotations: dict[str, tuple[float, float, float]] = field(default_factory=dict)


class LiveSegmentBuffer:
    """Collect frames between start/stop events and build a TrialRecord."""

    def __init__(self, fps: int, required_segments: tuple[str, ...]) -> None:
        self.fps = fps
        self.required_segments = required_segments
        self.frames: list[LiveFrame] = []

    def clear(self) -> None:
        """Drop all recorded frames."""
        self.frames.clear()

    def append(self, frame: LiveFrame) -> None:
        """Append one streamed frame."""
        self.frames.append(frame)

    def to_trial_record(
        self,
        trial_name: str = "live_segment",
        label: str = "unknown",
    ) -> TrialRecord:
        """Convert buffered frames to the project's trial representation."""
        if not self.frames:
            raise ValueError("Cannot build a trial from an empty live buffer.")

        missing = [
            segment_name
            for segment_name in self.required_segments
            if any(segment_name not in frame.translations for frame in self.frames)
        ]
        if missing:
            raise KeyError(f"Live segment is missing required segments: {missing}")

        first_frame = self.frames[0].frame_number
        last_frame = self.frames[-1].frame_number
        metadata = TrialMetadata(
            fps=self.fps,
            dt=1.0 / self.fps,
            num_frames=len(self.frames),
            frame_start=first_frame,
            frame_end=last_frame,
        )
        segments = {}
        for segment_name in self.required_segments:
            translation = np.asarray(
                [frame.translations[segment_name] for frame in self.frames],
                dtype=float,
            )
            rotation = np.asarray(
                [
                    frame.rotations.get(segment_name, (0.0, 0.0, 0.0))
                    for frame in self.frames
                ],
                dtype=float,
            )
            segments[segment_name] = SegmentTrajectory(
                name=segment_name,
                translation=translation,
                rotation=rotation,
            )

        return TrialRecord(
            path=Path("<live>"),
            patient="live",
            session="live",
            trial_name=trial_name,
            label=label,
            metadata=metadata,
            segments=segments,
        )
