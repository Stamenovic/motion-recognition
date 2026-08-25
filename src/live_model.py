"""Trainable models for segment-based live motion recognition."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid

from .classification import make_svm_pipeline
from .data_types import TrialRecord
from .feature_extraction import (
    LEFT_SEGMENT,
    RIGHT_SEGMENT,
    TRUNK_SEGMENT,
    extract_trial_features,
)
from .preprocessing import filter_trial_translations
from .temporal_normalization import normalize_trial_features


UNKNOWN_LABEL = "Nepoznato"
UNKNOWN_THRESHOLD_QUANTILE = 0.05
UNKNOWN_THRESHOLD_SCALE = 0.90
MIN_MOTION_QUANTILE = 0.05
MIN_MOTION_SCALE = 0.50
MOTION_SEGMENTS = (LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT)


@dataclass
class PredictionResult:
    """Prediction from the live-ready fPCA model."""

    fpca_prediction: str
    known_prediction: str
    fpca_confidence: float
    unknown_threshold: float
    motion_extent_mm: float
    minimum_motion_extent_mm: float
    is_unknown: bool


@dataclass
class LiveMotionModel:
    """Serializable fPCA/SVM model bundle."""

    cutoff_hz: float
    filter_order: int
    normalized_num_samples: int
    fpca_components: int
    signal_names: list[str]
    signal_means: dict[str, float]
    signal_stds: dict[str, float]
    fpca: FPCA
    fpca_svm: object
    unknown_threshold: float
    minimum_motion_extent_mm: float
    unknown_label: str
    labels: list[str]

    def save(self, output_path: Path) -> Path:
        """Persist the model bundle to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)
        return output_path

    @classmethod
    def load(cls, model_path: Path) -> "LiveMotionModel":
        """Load a persisted model bundle."""
        return joblib.load(model_path)

    def predict_trial(self, trial: TrialRecord) -> PredictionResult:
        """Predict a completed movement segment."""
        fpca_row = self.trial_to_fpca_row(trial)
        known_prediction = str(self.fpca_svm.predict(fpca_row)[0])
        confidence = _prediction_confidence(self.fpca_svm, fpca_row)
        motion_extent = _trial_motion_extent_mm(trial)
        is_unknown = (
            confidence < self.unknown_threshold
            or motion_extent < self.minimum_motion_extent_mm
        )
        fpca_prediction = self.unknown_label if is_unknown else known_prediction
        return PredictionResult(
            fpca_prediction=fpca_prediction,
            known_prediction=known_prediction,
            fpca_confidence=confidence,
            unknown_threshold=self.unknown_threshold,
            motion_extent_mm=motion_extent,
            minimum_motion_extent_mm=self.minimum_motion_extent_mm,
            is_unknown=is_unknown,
        )

    def trial_to_fpca_row(self, trial: TrialRecord) -> np.ndarray:
        """Convert one trial to one fPCA score row."""
        normalized = _preprocess_and_normalize_trial(
            trial,
            cutoff_hz=self.cutoff_hz,
            order=self.filter_order,
            num_samples=self.normalized_num_samples,
        )
        row = []
        for signal_name in self.signal_names:
            signal = normalized.signals[signal_name]
            mean = self.signal_means[signal_name]
            std = self.signal_stds[signal_name]
            if std == 0.0:
                row.append(signal - mean)
            else:
                row.append((signal - mean) / std)
        data_matrix = np.concatenate(row, dtype=float)[np.newaxis, :, np.newaxis]
        fd_grid = FDataGrid(
            data_matrix=data_matrix,
            grid_points=np.linspace(0.0, 1.0, data_matrix.shape[1]),
        )
        return np.asarray(self.fpca.transform(fd_grid), dtype=float)

def train_live_motion_model(
    trials: list[TrialRecord],
    cutoff_hz: float = 10.0,
    filter_order: int = 2,
    normalized_num_samples: int = 101,
    fpca_components: int = 3,
    unknown_label: str = UNKNOWN_LABEL,
) -> LiveMotionModel:
    """Train an fPCA/SVM model for completed live segments."""
    if len(trials) < 2:
        raise ValueError("At least two trials are required to train the model.")

    normalized_trials = [
        _preprocess_and_normalize_trial(
            trial,
            cutoff_hz=cutoff_hz,
            order=filter_order,
            num_samples=normalized_num_samples,
        )
        for trial in trials
    ]
    signal_names = sorted(normalized_trials[0].signals)
    signal_means, signal_stds = _fit_signal_standardization(
        normalized_trials,
        signal_names,
    )
    fpca_grid = _normalized_trials_to_standardized_fd_grid(
        normalized_trials,
        signal_names,
        signal_means,
        signal_stds,
    )
    max_components = min(fpca_components, len(trials) - 1)
    fpca = FPCA(n_components=max_components)
    fpca_scores = np.asarray(fpca.fit_transform(fpca_grid), dtype=float)
    labels = np.array([trial.label for trial in trials])
    fpca_svm = make_svm_pipeline()
    fpca_svm.fit(fpca_scores, labels)
    training_confidences = _prediction_confidences(fpca_svm, fpca_scores)
    unknown_threshold = float(
        np.quantile(training_confidences, UNKNOWN_THRESHOLD_QUANTILE)
        * UNKNOWN_THRESHOLD_SCALE
    )
    motion_extents = np.asarray(
        [_trial_motion_extent_mm(trial) for trial in trials],
        dtype=float,
    )
    minimum_motion_extent_mm = float(
        np.quantile(motion_extents, MIN_MOTION_QUANTILE) * MIN_MOTION_SCALE
    )

    return LiveMotionModel(
        cutoff_hz=cutoff_hz,
        filter_order=filter_order,
        normalized_num_samples=normalized_num_samples,
        fpca_components=max_components,
        signal_names=signal_names,
        signal_means=signal_means,
        signal_stds=signal_stds,
        fpca=fpca,
        fpca_svm=fpca_svm,
        unknown_threshold=unknown_threshold,
        minimum_motion_extent_mm=minimum_motion_extent_mm,
        unknown_label=unknown_label,
        labels=sorted(str(label) for label in set(labels)),
    )


def _prediction_confidence(model, rows: np.ndarray) -> float:
    """Return confidence for one row as the best SVM decision score."""
    return float(_prediction_confidences(model, rows)[0])


def _prediction_confidences(model, rows: np.ndarray) -> np.ndarray:
    """Return one confidence score per row from the SVM decision function."""
    scores = np.asarray(model.decision_function(rows), dtype=float)
    if scores.ndim == 1:
        return np.abs(scores)
    return np.max(scores, axis=1)


def _trial_motion_extent_mm(trial: TrialRecord) -> float:
    """Return the largest translation range among required movement segments."""
    extents = []
    for segment_name in MOTION_SEGMENTS:
        if segment_name not in trial.segments:
            continue
        translation = np.asarray(trial.segments[segment_name].translation, dtype=float)
        if translation.size == 0 or not np.isfinite(translation).any():
            continue
        axis_range = np.nanmax(translation, axis=0) - np.nanmin(translation, axis=0)
        extents.append(float(np.linalg.norm(axis_range)))
    return max(extents, default=0.0)


def _preprocess_and_normalize_trial(
    trial: TrialRecord,
    cutoff_hz: float,
    order: int,
    num_samples: int,
):
    filtered = filter_trial_translations(trial, cutoff_hz=cutoff_hz, order=order)
    features = extract_trial_features(filtered)
    return normalize_trial_features(features, num_samples=num_samples)


def _fit_signal_standardization(normalized_trials, signal_names):
    means = {}
    stds = {}
    for signal_name in signal_names:
        stack = np.array(
            [trial.signals[signal_name] for trial in normalized_trials],
            dtype=float,
        )
        means[signal_name] = float(np.mean(stack))
        stds[signal_name] = float(np.std(stack))
    return means, stds


def _normalized_trials_to_standardized_fd_grid(
    normalized_trials,
    signal_names,
    means,
    stds,
) -> FDataGrid:
    rows = []
    for trial in normalized_trials:
        parts = []
        for signal_name in signal_names:
            signal = trial.signals[signal_name]
            std = stds[signal_name]
            if std == 0.0:
                parts.append(signal - means[signal_name])
            else:
                parts.append((signal - means[signal_name]) / std)
        rows.append(np.concatenate(parts))
    data_matrix = np.asarray(rows, dtype=float)[:, :, np.newaxis]
    return FDataGrid(
        data_matrix=data_matrix,
        grid_points=np.linspace(0.0, 1.0, data_matrix.shape[1]),
    )
