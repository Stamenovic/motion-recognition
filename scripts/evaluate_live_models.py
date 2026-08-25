"""Evaluate live-ready models with Leave-One-Out validation."""
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import RAW_DATA_DIR
from src.data_loader import load_trials
from src.live_model import UNKNOWN_LABEL, train_live_motion_model


def main() -> None:
    trials = load_trials(RAW_DATA_DIR)
    if len(trials) < 2:
        raise RuntimeError("At least two trials are required for Leave-One-Out.")

    y_true = []
    fpca_predictions = []
    trial_names = []

    for train_idx, test_idx in LeaveOneOut().split(np.arange(len(trials))):
        train_trials = [trials[index] for index in train_idx]
        test_trial = trials[test_idx[0]]
        model = train_live_motion_model(train_trials)
        result = model.predict_trial(test_trial)

        trial_names.append(test_trial.trial_name)
        y_true.append(test_trial.label)
        fpca_predictions.append(result.fpca_prediction)

    labels = sorted(set(y_true) | set(fpca_predictions) | {UNKNOWN_LABEL})
    print("Live-ready Leave-One-Out evaluation")
    print()
    _print_result("fPCA + SVM", y_true, fpca_predictions, labels, trial_names)


def _print_result(name, y_true, y_pred, labels, trial_names) -> None:
    print(name)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels))
    for trial_name, truth, prediction in zip(trial_names, y_true, y_pred):
        status = "OK" if truth == prediction else "MISS"
        print(f"  {trial_name}: true={truth}, predicted={prediction} [{status}]")


if __name__ == "__main__":
    main()
