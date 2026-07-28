"""SVM classification experiments for fixed-length motion features."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .statistical_features import StatisticalTrialFeatures


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


def build_statistical_dataset(
    statistical_trials: list[StatisticalTrialFeatures],
) -> ClassificationDataset:
    """Build X/y arrays from non-normalized statistical trial features."""
    if not statistical_trials:
        raise ValueError("No statistical trials provided.")

    feature_names = sorted(statistical_trials[0].values)
    rows = []
    labels = []
    trial_names = []

    for trial_features in statistical_trials:
        rows.append([trial_features.values[name] for name in feature_names])
        labels.append(trial_features.label)
        trial_names.append(trial_features.trial_name)

    return ClassificationDataset(
        X=np.array(rows, dtype=float),
        y=np.array(labels),
        trial_names=trial_names,
        feature_names=feature_names,
    )


def make_svm_pipeline() -> Pipeline:
    """Create the default SVM classifier pipeline."""
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

    loo = LeaveOneOut()
    predictions = []
    truths = []
    predicted_trial_names = []

    for train_idx, test_idx in loo.split(dataset.X, dataset.y):
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
