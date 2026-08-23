"""Trainable models for segment-based live motion recognition."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.grid import FDataGrid

from .classification import make_svm_pipeline
from .data_types import TrialRecord
from .feature_extraction import extract_trial_features
from .preprocessing import filter_trial_translations
from .statistical_features import extract_statistical_features
from .temporal_normalization import normalize_trial_features


@dataclass
class PredictionResult:
    """Predictions from the live-ready model pair."""

    fpca_prediction: str
    statistical_prediction: str
    models_agree: bool


@dataclass
class LiveMotionModel:
    """Serializable fPCA/SVM and statistical/SVM model bundle."""

    cutoff_hz: float
    filter_order: int
    normalized_num_samples: int
    fpca_components: int
    signal_names: list[str]
    signal_means: dict[str, float]
    signal_stds: dict[str, float]
    fpca: FPCA
    fpca_svm: object
    statistical_feature_names: list[str]
    statistical_svm: object
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
        statistical_row = self.trial_to_statistical_row(trial)
        fpca_prediction = str(self.fpca_svm.predict(fpca_row)[0])
        statistical_prediction = str(self.statistical_svm.predict(statistical_row)[0])
        return PredictionResult(
            fpca_prediction=fpca_prediction,
            statistical_prediction=statistical_prediction,
            models_agree=fpca_prediction == statistical_prediction,
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

    def trial_to_statistical_row(self, trial: TrialRecord) -> np.ndarray:
        """Convert one trial to one statistical feature row."""
        filtered = filter_trial_translations(
            trial,
            cutoff_hz=self.cutoff_hz,
            order=self.filter_order,
        )
        features = extract_trial_features(filtered)
        statistical = extract_statistical_features(features)
        return np.array(
            [[statistical.values[name] for name in self.statistical_feature_names]],
            dtype=float,
        )


def train_live_motion_model(
    trials: list[TrialRecord],
    cutoff_hz: float = 10.0,
    filter_order: int = 2,
    normalized_num_samples: int = 101,
    fpca_components: int = 2,
) -> LiveMotionModel:
    """Train fPCA/SVM and statistical/SVM models for completed live segments."""
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

    statistical_trials = [
        extract_statistical_features(
            extract_trial_features(
                filter_trial_translations(
                    trial,
                    cutoff_hz=cutoff_hz,
                    order=filter_order,
                )
            )
        )
        for trial in trials
    ]
    statistical_feature_names = sorted(statistical_trials[0].values)
    statistical_matrix = np.array(
        [
            [trial.values[name] for name in statistical_feature_names]
            for trial in statistical_trials
        ],
        dtype=float,
    )
    statistical_svm = make_svm_pipeline()
    statistical_svm.fit(statistical_matrix, labels)

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
        statistical_feature_names=statistical_feature_names,
        statistical_svm=statistical_svm,
        labels=sorted(str(label) for label in set(labels)),
    )


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
