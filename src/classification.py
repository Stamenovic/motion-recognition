"""SVM classification experiments for motion features."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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


def flatten_normalized_features(
    normalized_features: NormalizedTrialFeatures,
    feature_names: list[str] | None = None,
    scalar_feature_names: list[str] | None = None,
    include_scalar_features: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Flatten normalized feature signals into one SVM input vector."""
    names = feature_names or sorted(normalized_features.signals)
    signal_vector = np.concatenate([normalized_features.signals[name] for name in names])
    vector_parts = [signal_vector]
    expanded_names = [
        f"{name}[{sample_idx:03d}]"
        for name in names
        for sample_idx in range(len(normalized_features.signals[name]))
    ]

    if include_scalar_features:
        scalar_names = scalar_feature_names or sorted(normalized_features.scalar_features)
        scalar_vector = np.array(
            [normalized_features.scalar_features[name] for name in scalar_names],
            dtype=float,
        )
        vector_parts.append(scalar_vector)
        expanded_names.extend(scalar_names)

    vector = np.concatenate(vector_parts)
    return vector, expanded_names


def build_normalized_dataset(
    normalized_trials: list[NormalizedTrialFeatures],
    include_scalar_features: bool = True,
) -> ClassificationDataset:
    """Build X/y arrays from temporally normalized signal features."""
    if not normalized_trials:
        raise ValueError("No normalized trials provided.")

    signal_names = sorted(normalized_trials[0].signals)
    scalar_feature_names = sorted(normalized_trials[0].scalar_features)
    rows = []
    labels = []
    trial_names = []

    for trial_features in normalized_trials:
        vector, expanded_names = flatten_normalized_features(
            trial_features,
            feature_names=signal_names,
            scalar_feature_names=scalar_feature_names,
            include_scalar_features=include_scalar_features,
        )
        rows.append(vector)
        labels.append(trial_features.label)
        trial_names.append(trial_features.trial_name)

    return ClassificationDataset(
        X=np.vstack(rows),
        y=np.array(labels),
        trial_names=trial_names,
        feature_names=expanded_names,
    )


def build_scalar_feature_dataset(
    normalized_trials: list[NormalizedTrialFeatures],
    scalar_feature_names: list[str] | None = None,
) -> ClassificationDataset:
    """Build X/y arrays using only scalar trend features."""
    if not normalized_trials:
        raise ValueError("No normalized trials provided.")

    names = scalar_feature_names or sorted(normalized_trials[0].scalar_features)
    rows = []
    labels = []
    trial_names = []

    for trial_features in normalized_trials:
        rows.append([trial_features.scalar_features[name] for name in names])
        labels.append(trial_features.label)
        trial_names.append(trial_features.trial_name)

    return ClassificationDataset(
        X=np.array(rows, dtype=float),
        y=np.array(labels),
        trial_names=trial_names,
        feature_names=names,
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
