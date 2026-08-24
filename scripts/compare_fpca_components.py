"""Compare live fPCA/SVM performance for several component counts."""
import csv
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

from config import PLOTS_DIR, RAW_DATA_DIR, create_project_directories
from src.data_loader import load_trials
from src.live_model import train_live_motion_model


COMPONENT_COUNTS = (2, 3, 4, 5)
CLASS_LABELS = ("guranje", "podizanje_desna", "sirenje")
OUTPUT_DIR = PLOTS_DIR / "live_generalization"


def group_name(path: Path) -> str | None:
    """Return data owner group for data2026 recordings."""
    lower_path = str(path).lower()
    if "data2026" not in lower_path:
        return None
    if "lazar" in lower_path:
        return "lazar"
    if "petar" in lower_path:
        return "petar"
    if "anja" in lower_path:
        return "anja"
    return None


def split_trials():
    """Use Anja+Petar for training, with one held-out trial per class."""
    trials = load_trials(RAW_DATA_DIR)
    anja_petar = sorted(
        [trial for trial in trials if group_name(trial.path) in ("anja", "petar")],
        key=lambda trial: (trial.label, str(trial.path)),
    )
    lazar = sorted(
        [trial for trial in trials if group_name(trial.path) == "lazar"],
        key=lambda trial: (trial.label, str(trial.path)),
    )

    train = []
    heldout = []
    for label in CLASS_LABELS:
        label_trials = [trial for trial in anja_petar if trial.label == label]
        if len(label_trials) < 2:
            raise RuntimeError(f"Need at least two Anja/Petar trials for {label}.")
        heldout.append(label_trials[-1])
        train.extend(label_trials[:-1])
    return train, heldout, lazar


def evaluate(model, trials) -> float:
    """Return accuracy for one trained live model on a list of trials."""
    y_true = [trial.label for trial in trials]
    y_pred = [model.predict_trial(trial).fpca_prediction for trial in trials]
    return float(accuracy_score(y_true, y_pred))


def run_comparison():
    """Train and evaluate fPCA/SVM models with 2, 3, 4 and 5 components."""
    train, heldout, lazar = split_trials()
    rows = []
    for requested_components in COMPONENT_COUNTS:
        model = train_live_motion_model(train, fpca_components=requested_components)
        variance_ratio = np.asarray(model.fpca.explained_variance_ratio_, dtype=float)
        rows.append(
            {
                "requested_components": requested_components,
                "actual_components": model.fpca_components,
                "heldout_accuracy": evaluate(model, heldout),
                "lazar_accuracy": evaluate(model, lazar),
                "explained_variance_ratio": variance_ratio,
                "cumulative_explained_variance": float(np.sum(variance_ratio)),
            }
        )
    return train, heldout, lazar, rows


def save_csv(rows) -> Path:
    """Save numeric comparison results."""
    output_path = OUTPUT_DIR / "fpca_component_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "requested_components",
                "actual_components",
                "heldout_accuracy",
                "lazar_accuracy",
                "cumulative_explained_variance",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["requested_components"],
                    row["actual_components"],
                    f"{row['heldout_accuracy']:.6f}",
                    f"{row['lazar_accuracy']:.6f}",
                    f"{row['cumulative_explained_variance']:.6f}",
                ]
            )
    return output_path


def plot_accuracy(rows) -> Path:
    """Plot held-out and Lazar accuracy by fPCA component count."""
    output_path = OUTPUT_DIR / "06_fpca_component_accuracy.png"
    components = [row["actual_components"] for row in rows]
    heldout = [row["heldout_accuracy"] for row in rows]
    lazar = [row["lazar_accuracy"] for row in rows]

    x = np.arange(len(components))
    width = 0.36
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.bar(x - width / 2, heldout, width, label="Anja+Petar held-out")
    axis.bar(x + width / 2, lazar, width, label="Lazar external")
    axis.set_title("Accuracy by fPCA component count")
    axis.set_xlabel("fPCA components")
    axis.set_ylabel("Accuracy")
    axis.set_xticks(x, components)
    axis.set_ylim(0.0, 1.08)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    for idx, (heldout_value, lazar_value) in enumerate(zip(heldout, lazar)):
        axis.text(idx - width / 2, heldout_value, f"{heldout_value:.3f}", ha="center", va="bottom")
        axis.text(idx + width / 2, lazar_value, f"{lazar_value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_variance(rows) -> Path:
    """Plot cumulative explained variance by selected fPCA component count."""
    output_path = OUTPUT_DIR / "07_fpca_component_variance.png"
    components = [row["actual_components"] for row in rows]
    cumulative = [row["cumulative_explained_variance"] for row in rows]

    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(components, cumulative, marker="o", linewidth=2.0)
    axis.set_title("Cumulative explained variance by fPCA component count")
    axis.set_xlabel("fPCA components")
    axis.set_ylabel("Cumulative explained variance ratio")
    axis.set_xticks(components)
    axis.set_ylim(0.0, min(1.05, max(cumulative) + 0.1))
    axis.grid(alpha=0.25)
    for x_value, y_value in zip(components, cumulative):
        axis.text(x_value, y_value, f"{y_value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_accuracy_and_variance(rows) -> Path:
    """Plot accuracy and cumulative explained variance in one figure."""
    output_path = OUTPUT_DIR / "08_fpca_accuracy_and_variance.png"
    components = [row["actual_components"] for row in rows]
    heldout = [row["heldout_accuracy"] for row in rows]
    lazar = [row["lazar_accuracy"] for row in rows]
    cumulative = [row["cumulative_explained_variance"] for row in rows]

    fig, (accuracy_axis, variance_axis) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12, 4.8),
    )

    x = np.arange(len(components))
    width = 0.36
    accuracy_axis.bar(x - width / 2, heldout, width, label="Anja+Petar held-out")
    accuracy_axis.bar(x + width / 2, lazar, width, label="Lazar external")
    accuracy_axis.set_title("Accuracy")
    accuracy_axis.set_xlabel("fPCA components")
    accuracy_axis.set_ylabel("Accuracy")
    accuracy_axis.set_xticks(x, components)
    accuracy_axis.set_ylim(0.0, 1.08)
    accuracy_axis.grid(axis="y", alpha=0.25)
    accuracy_axis.legend(loc="lower right")
    for idx, (heldout_value, lazar_value) in enumerate(zip(heldout, lazar)):
        accuracy_axis.text(
            idx - width / 2,
            heldout_value,
            f"{heldout_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        accuracy_axis.text(
            idx + width / 2,
            lazar_value,
            f"{lazar_value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    variance_axis.plot(components, cumulative, marker="o", linewidth=2.0)
    variance_axis.axvline(3, color="tab:red", linestyle="--", linewidth=1.2, label="chosen: 3")
    variance_axis.set_title("Cumulative explained variance")
    variance_axis.set_xlabel("fPCA components")
    variance_axis.set_ylabel("Variance ratio")
    variance_axis.set_xticks(components)
    variance_axis.set_ylim(0.0, min(1.05, max(cumulative) + 0.1))
    variance_axis.grid(alpha=0.25)
    variance_axis.legend(loc="lower right")
    for x_value, y_value in zip(components, cumulative):
        variance_axis.text(x_value, y_value, f"{y_value:.3f}", ha="center", va="bottom")

    fig.suptitle("fPCA component-count comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    create_project_directories()
    train, heldout, lazar, rows = run_comparison()
    print(f"Train: {len(train)} {dict(sorted(Counter(trial.label for trial in train).items()))}")
    print(f"Held-out: {len(heldout)} {dict(sorted(Counter(trial.label for trial in heldout).items()))}")
    print(f"Lazar: {len(lazar)} {dict(sorted(Counter(trial.label for trial in lazar).items()))}")
    for row in rows:
        print(
            f"components={row['actual_components']} "
            f"heldout={row['heldout_accuracy']:.3f} "
            f"lazar={row['lazar_accuracy']:.3f} "
            f"cum_var={row['cumulative_explained_variance']:.3f}"
        )

    for path in (
        save_csv(rows),
        plot_accuracy(rows),
        plot_variance(rows),
        plot_accuracy_and_variance(rows),
    ):
        print(path)


if __name__ == "__main__":
    main()
