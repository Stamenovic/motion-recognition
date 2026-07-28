"""Temporal normalization for extracted movement feature signals."""
from dataclasses import dataclass

import numpy as np

from .feature_extraction import TrialFeatures


DEFAULT_NUM_SAMPLES = 101
TREND_SIGNAL_NAME = "distance_left_right"


@dataclass
class NormalizedTrialFeatures:
    """Feature signals resampled to a common normalized time base."""
    trial_name: str
    label: str
    normalized_time: np.ndarray
    signals: dict[str, np.ndarray]
    scalar_features: dict[str, float]
    original_duration: float
    original_num_frames: int


def resample_signal(signal: np.ndarray, num_samples: int = DEFAULT_NUM_SAMPLES) -> np.ndarray:
    """Resample one 1D signal to a fixed number of uniformly spaced samples."""
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}.")
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2.")

    original_time = np.linspace(0.0, 1.0, signal.shape[0])
    normalized_time = np.linspace(0.0, 1.0, num_samples)
    return np.interp(normalized_time, original_time, signal)


def extract_trend_features(
    signal: np.ndarray,
    normalized_time: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    """Extract scalar trend descriptors from one normalized 1D signal."""
    if signal.ndim != 1:
        raise ValueError(f"Expected a 1D signal, got shape {signal.shape}.")
    if len(signal) != len(normalized_time):
        raise ValueError("signal and normalized_time must have the same length.")

    differences = np.diff(signal)
    slope = np.gradient(signal, normalized_time)
    positive_tolerance = 1e-9
    max_idx = int(np.argmax(signal))
    min_idx = int(np.argmin(signal))

    return {
        f"{prefix}_start": float(signal[0]),
        f"{prefix}_end": float(signal[-1]),
        f"{prefix}_delta": float(signal[-1] - signal[0]),
        f"{prefix}_min": float(signal[min_idx]),
        f"{prefix}_max": float(signal[max_idx]),
        f"{prefix}_range": float(signal[max_idx] - signal[min_idx]),
        f"{prefix}_mean_slope": float(np.mean(slope)),
        f"{prefix}_positive_slope_ratio": float(
            np.mean(differences > positive_tolerance)
        ),
        f"{prefix}_negative_slope_ratio": float(
            np.mean(differences < -positive_tolerance)
        ),
        f"{prefix}_time_to_max": float(normalized_time[max_idx]),
        f"{prefix}_time_to_min": float(normalized_time[min_idx]),
        f"{prefix}_auc": float(np.trapezoid(signal, normalized_time)),
    }


def normalize_trial_features(
    features: TrialFeatures,
    num_samples: int = DEFAULT_NUM_SAMPLES,
) -> NormalizedTrialFeatures:
    """Resample all feature signals to a common normalized time base."""
    normalized_time = np.linspace(0.0, 1.0, num_samples)
    normalized_signals = {
        name: resample_signal(signal, num_samples=num_samples)
        for name, signal in features.signals.items()
    }
    scalar_features = extract_trend_features(
        normalized_signals[TREND_SIGNAL_NAME],
        normalized_time,
        prefix=TREND_SIGNAL_NAME,
    )

    return NormalizedTrialFeatures(
        trial_name=features.trial_name,
        label=features.label,
        normalized_time=normalized_time,
        signals=normalized_signals,
        scalar_features=scalar_features,
        original_duration=float(features.time[-1]) if len(features.time) else 0.0,
        original_num_frames=len(features.time),
    )
