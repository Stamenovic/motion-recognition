"""Fixed-length statistical summaries for non-normalized feature signals."""
from dataclasses import dataclass

import numpy as np

from .feature_extraction import TrialFeatures


@dataclass
class StatisticalTrialFeatures:
    """Scalar feature representation of one trial without temporal resampling."""
    trial_name: str
    label: str
    values: dict[str, float]


def summarize_signal(signal: np.ndarray, time: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize one variable-length signal into fixed scalar descriptors."""
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if len(signal) != len(time):
        raise ValueError("signal and time must have the same length.")

    differences = np.diff(signal)
    duration = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    max_idx = int(np.argmax(signal))
    min_idx = int(np.argmin(signal))

    if duration > 0:
        mean_slope = float((signal[-1] - signal[0]) / duration)
        auc = float(np.trapezoid(signal, time) / duration)
    else:
        mean_slope = 0.0
        auc = float(signal[0]) if len(signal) else 0.0

    return {
        f"{prefix}_mean": float(np.mean(signal)),
        f"{prefix}_std": float(np.std(signal)),
        f"{prefix}_min": float(signal[min_idx]),
        f"{prefix}_max": float(signal[max_idx]),
        f"{prefix}_range": float(signal[max_idx] - signal[min_idx]),
        f"{prefix}_median": float(np.median(signal)),
        f"{prefix}_p25": float(np.percentile(signal, 25)),
        f"{prefix}_p75": float(np.percentile(signal, 75)),
        f"{prefix}_start": float(signal[0]),
        f"{prefix}_end": float(signal[-1]),
        f"{prefix}_delta": float(signal[-1] - signal[0]),
        f"{prefix}_mean_slope": mean_slope,
        f"{prefix}_positive_slope_ratio": float(np.mean(differences > 1e-9)),
        f"{prefix}_time_to_max_ratio": float(max_idx / max(len(signal) - 1, 1)),
        f"{prefix}_time_to_min_ratio": float(min_idx / max(len(signal) - 1, 1)),
        f"{prefix}_auc_per_second": auc,
    }


def extract_statistical_features(features: TrialFeatures) -> StatisticalTrialFeatures:
    """Convert variable-length time-series features into scalar statistics."""
    values: dict[str, float] = {
        "trial_duration_seconds": float(features.time[-1]) if len(features.time) else 0.0,
        "trial_num_frames": float(len(features.time)),
    }

    for signal_name, signal in features.signals.items():
        values.update(summarize_signal(signal, features.time, prefix=signal_name))

    return StatisticalTrialFeatures(
        trial_name=features.trial_name,
        label=features.label,
        values=values,
    )
