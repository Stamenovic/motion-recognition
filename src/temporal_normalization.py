"""Temporal normalization for extracted movement feature signals."""
from dataclasses import dataclass

import numpy as np

from .feature_extraction import TrialFeatures


DEFAULT_NUM_SAMPLES = 101


@dataclass
class NormalizedTrialFeatures:
    """Feature signals resampled to a common normalized time base."""
    trial_name: str
    label: str
    normalized_time: np.ndarray
    signals: dict[str, np.ndarray]
    original_duration: float
    original_num_frames: int


def resample_signal(signal: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES) -> np.ndarray:
    """Resample one 1D signal to a fixed number of uniformly spaced samples."""
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}.")
    original_time = np.linspace(0.0, 1.0, signal.shape[0])
    normalized_time = np.linspace(0.0, 1.0, num_samples)
    return np.interp(normalized_time, original_time, signal)


def normalize_trial_features(
    features: TrialFeatures,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> NormalizedTrialFeatures:
    """Resample all feature signals to a common normalized time base."""
    normalized_time = np.linspace(0.0, 1.0, num_samples)
    return NormalizedTrialFeatures(
        trial_name=features.trial_name,
        label=features.label,
        normalized_time=normalized_time,
        signals={
            name: resample_signal(signal, num_samples=num_samples)
            for name, signal in features.signals.items()
        },
        original_duration=float(features.time[-1]) if len(features.time) else 0.0,
        original_num_frames=len(features.time),
    )
