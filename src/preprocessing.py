"""Preprocessing utilities for Vicon segment trajectories."""
from dataclasses import replace

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data_types import SegmentTrajectory, TrialRecord


def fill_missing_values(signal: np.ndarray) -> np.ndarray:
    """Fill NaN samples column-wise by linear interpolation."""
    filled = np.asarray(signal, dtype=float).copy()
    if filled.ndim != 2:
        raise ValueError(f"Expected a 2D signal, got shape {filled.shape}.")

    x = np.arange(filled.shape[0])
    for column_idx in range(filled.shape[1]):
        column = filled[:, column_idx]
        valid = np.isfinite(column)
        if valid.all():
            continue
        if not valid.any():
            filled[:, column_idx] = 0.0
            continue
        filled[:, column_idx] = np.interp(x, x[valid], column[valid])
    return filled


def butterworth_lowpass(
    signal: np.ndarray,
    fps: int,
    cutoff_hz: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter along the time axis."""
    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")
    nyquist_hz = fps / 2.0
    if cutoff_hz >= nyquist_hz:
        raise ValueError(f"cutoff_hz must be lower than {nyquist_hz} Hz.")

    clean_signal = fill_missing_values(signal)
    if clean_signal.shape[0] <= order * 3:
        return clean_signal

    sos = butter(order, cutoff_hz, btype="lowpass", fs=fps, output="sos")
    return sosfiltfilt(sos, clean_signal, axis=0)


def filter_segment_translation(
    segment: SegmentTrajectory,
    fps: int,
    cutoff_hz: float = 10.0,
    order: int = 4,
) -> SegmentTrajectory:
    """Return a copy of a segment with filtered translation data."""
    return SegmentTrajectory(
        name=segment.name,
        translation=butterworth_lowpass(segment.translation, fps, cutoff_hz, order),
        rotation=segment.rotation.copy(),
    )


def filter_trial_translations(
    trial: TrialRecord,
    cutoff_hz: float = 10.0,
    order: int = 4,
) -> TrialRecord:
    """Return a copy of a trial with all segment translations low-pass filtered."""
    return replace(
        trial,
        segments={
            name: filter_segment_translation(segment, trial.metadata.fps, cutoff_hz, order)
            for name, segment in trial.segments.items()
        },
    )
