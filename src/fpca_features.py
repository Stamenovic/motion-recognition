"""Functional PCA feature extraction for normalized motion signals."""
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid

from .temporal_normalization import NormalizedTrialFeatures


@dataclass
class FPCADataset:
    """fPCA scores with labels and diagnostic metadata."""
    X: np.ndarray
    y: np.ndarray
    trial_names: list[str]
    component_names: list[str]
    explained_variance_ratio: np.ndarray
    signal_names: list[str]
    standardized: bool


def stack_signal(
    normalized_trials: list[NormalizedTrialFeatures],
    signal_name: str,
) -> np.ndarray:
    """Stack one normalized signal into a trials x time array."""
    return np.array([trial.signals[signal_name] for trial in normalized_trials], dtype=float)


def standardize_signal_stack(signal_stack: np.ndarray) -> np.ndarray:
    """Standardize one signal across all trials and time samples."""
    mean = float(np.mean(signal_stack))
    std = float(np.std(signal_stack))
    if std == 0.0:
        return signal_stack - mean
    return (signal_stack - mean) / std


def normalized_trials_to_fd_grid(
    normalized_trials: list[NormalizedTrialFeatures],
    signal_names: list[str] | None = None,
    standardize_signals: bool = True,
) -> tuple[FDataGrid, list[str], list[str], np.ndarray]:
    """Convert normalized signals to a scalar-valued FDataGrid for fPCA."""
    if not normalized_trials:
        raise ValueError("No normalized trials provided.")

    names = signal_names or sorted(normalized_trials[0].signals)
    signal_stacks = {name: stack_signal(normalized_trials, name) for name in names}
    if standardize_signals:
        signal_stacks = {
            name: standardize_signal_stack(stack)
            for name, stack in signal_stacks.items()
        }

    num_points = sum(len(normalized_trials[0].signals[name]) for name in names)
    grid_points = np.linspace(0.0, 1.0, num_points)
    data_matrix = np.array(
        [
            np.concatenate([signal_stacks[name][trial_idx] for name in names])
            for trial_idx in range(len(normalized_trials))
        ],
        dtype=float,
    )[:, :, np.newaxis]
    fd_grid = FDataGrid(data_matrix=data_matrix, grid_points=grid_points)
    labels = [trial.label for trial in normalized_trials]
    trial_names = [trial.trial_name for trial in normalized_trials]
    return fd_grid, labels, trial_names, names


def build_fpca_dataset(
    normalized_trials: list[NormalizedTrialFeatures],
    n_components: int = 1,
    signal_names: list[str] | None = None,
    standardize_signals: bool = True,
) -> FPCADataset:
    """Fit fPCA and return component scores as an SVM-ready dataset."""
    fd_grid, labels, trial_names, names = normalized_trials_to_fd_grid(
        normalized_trials,
        signal_names=signal_names,
        standardize_signals=standardize_signals,
    )
    max_components = min(n_components, len(normalized_trials) - 1)
    if max_components < 1:
        raise ValueError("At least two trials are required for fPCA.")

    fpca = FPCA(n_components=max_components)
    scores = fpca.fit_transform(fd_grid)
    explained = getattr(fpca, "explained_variance_ratio_", np.array([]))

    return FPCADataset(
        X=np.asarray(scores, dtype=float),
        y=np.array(labels),
        trial_names=trial_names,
        component_names=[f"fPC{i + 1}" for i in range(max_components)],
        explained_variance_ratio=np.asarray(explained, dtype=float),
        signal_names=names,
        standardized=standardize_signals,
    )
