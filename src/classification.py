"""SVM classification experiments for motion features."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .statistical_features import StatisticalTrialFeatures
from .temporal_normalization import NormalizedTrialFeatures


@dataclass
class ClassificationDataset:
    """Matrix representation of trial features for scikit-learn classifiers."""
    X: np.ndarray
    y: np.ndarray
    trial_names: list[str]
    feature_names: list[str]


@dataclass
class ClassificationResult:
    """Cross-validated classification result."""
    y_true: np.ndarray
    y_pred: np.ndarray
    labels: list[str]
    trial_names: list[str]
    accuracy: float
    confusion: np.ndarray


def make_svm_pipeline() -> Pipeline:
    """Create a linear SVM with feature standardization inside each CV split."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", C=1.0)),
        ]
    )


def evaluate_leave_one_out_svm(dataset: ClassificationDataset) -> ClassificationResult:
    """Evaluate a linear SVM using Leave-One-Out cross-validation."""
    if len(dataset.y) < 2:
        raise ValueError("At least two trials are required for classification.")

    predictions = []
    truths = []
    predicted_trial_names = []

    for train_idx, test_idx in LeaveOneOut().split(dataset.X, dataset.y):
        model = make_svm_pipeline()
        model.fit(dataset.X[train_idx], dataset.y[train_idx])
        prediction = model.predict(dataset.X[test_idx])
        predictions.append(prediction[0])
        truths.append(dataset.y[test_idx][0])
        predicted_trial_names.append(dataset.trial_names[test_idx[0]])

    labels = sorted(str(label) for label in set(dataset.y))
    y_true = np.array(truths)
    y_pred = np.array(predictions)
    return ClassificationResult(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        trial_names=predicted_trial_names,
        accuracy=float(accuracy_score(y_true, y_pred)),
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def build_normalized_signal_dataset(
    normalized_trials: list[NormalizedTrialFeatures],
) -> ClassificationDataset:
    """Flatten normalized time-series signals into one SVM row per trial."""
    if not normalized_trials:
        raise ValueError("No normalized trials provided.")

    signal_names = sorted(normalized_trials[0].signals)
    feature_names = [
        f"{signal_name}_t{sample_idx:03d}"
        for signal_name in signal_names
        for sample_idx in range(len(normalized_trials[0].signals[signal_name]))
    ]
    X = np.array(
        [
            np.concatenate([trial.signals[signal_name] for signal_name in signal_names])
            for trial in normalized_trials
        ],
        dtype=float,
    )
    return ClassificationDataset(
        X=X,
        y=np.array([trial.label for trial in normalized_trials]),
        trial_names=[trial.trial_name for trial in normalized_trials],
        feature_names=feature_names,
    )


def build_statistical_dataset(
    statistical_trials: list[StatisticalTrialFeatures],
) -> ClassificationDataset:
    """Convert scalar statistical features into an SVM matrix."""
    if not statistical_trials:
        raise ValueError("No statistical trials provided.")

    feature_names = sorted(statistical_trials[0].values)
    X = np.array(
        [[trial.values[name] for name in feature_names] for trial in statistical_trials],
        dtype=float,
    )
    return ClassificationDataset(
        X=X,
        y=np.array([trial.label for trial in statistical_trials]),
        trial_names=[trial.trial_name for trial in statistical_trials],
        feature_names=feature_names,
    )
