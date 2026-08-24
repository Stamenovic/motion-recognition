"""Generate plots for Anja+Petar training and Lazar external testing."""
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

import matplotlib.pyplot as plt

from config import PLOTS_DIR, RAW_DATA_DIR, create_project_directories
from src.data_loader import load_trials
from src.live_model import LiveMotionModel, train_live_motion_model


OUTPUT_DIR = PLOTS_DIR / "live_generalization"
CLASS_LABELS = ("guranje", "podizanje_desna", "sirenje")
CLASS_COLORS = {
    "guranje": "tab:blue",
    "podizanje_desna": "tab:green",
    "sirenje": "tab:orange",
}
SPLIT_MARKERS = {
    "train": "o",
    "anja_petar_test": "s",
    "lazar_external": "^",
}


@dataclass
class PredictionRow:
    trial_name: str
    relative_path: str
    split: str
    true_label: str
    predicted_label: str
    fpca_scores: np.ndarray


def group_name(path: Path) -> str | None:
    """Return the data owner group inferred from a data2026 path."""
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
    """Train on Anja+Petar, hold out one Anja/Petar trial per class, test Lazar."""
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


def predict_trials(model: LiveMotionModel, trials, split_name: str) -> list[PredictionRow]:
    """Predict trials and keep fPCA scores for plotting."""
    rows = []
    for trial in trials:
        fpca_scores = model.trial_to_fpca_row(trial)[0]
        prediction = model.predict_trial(trial).fpca_prediction
        rows.append(
            PredictionRow(
                trial_name=trial.trial_name,
                relative_path=str(trial.path.relative_to(RAW_DATA_DIR)),
                split=split_name,
                true_label=trial.label,
                predicted_label=prediction,
                fpca_scores=fpca_scores,
            )
        )
    return rows


def finish(fig, output_path: Path) -> Path:
    """Save a plot with consistent output settings."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_split_counts(train, heldout, lazar) -> Path:
    """Show how many trials are used in each split and class."""
    split_counts = {
        "Train\nAnja+Petar": Counter(trial.label for trial in train),
        "Held-out\nAnja+Petar": Counter(trial.label for trial in heldout),
        "External\nLazar": Counter(trial.label for trial in lazar),
    }
    fig, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(len(split_counts))
    bottom = np.zeros(len(split_counts))
    for label in CLASS_LABELS:
        values = [split_counts[split][label] for split in split_counts]
        bars = axis.bar(
            x,
            values,
            bottom=bottom,
            color=CLASS_COLORS[label],
            label=label,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    str(value),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontweight="bold",
                )
        bottom += values
    axis.set_title("Dataset split used for live generalization check")
    axis.set_ylabel("Number of trials")
    axis.set_xticks(x, split_counts)
    axis.legend(loc="upper right")
    axis.grid(axis="y", alpha=0.25)
    return finish(fig, OUTPUT_DIR / "01_dataset_split_counts.png")


def plot_confusion(rows: list[PredictionRow], title: str, file_name: str) -> Path:
    """Plot confusion matrix for one evaluation split."""
    y_true = [row.true_label for row in rows]
    y_pred = [row.predicted_label for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_LABELS))
    accuracy = accuracy_score(y_true, y_pred)

    fig, axis = plt.subplots(figsize=(6.3, 5.4))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set_title(f"{title} | accuracy={accuracy:.3f}")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(CLASS_LABELS)), CLASS_LABELS, rotation=25, ha="right")
    axis.set_yticks(range(len(CLASS_LABELS)), CLASS_LABELS)

    max_value = matrix.max() if matrix.size else 0
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            color = "white" if value > max_value / 2 else "black"
            axis.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)
    return finish(fig, OUTPUT_DIR / file_name)


def plot_fpca_scores(rows: list[PredictionRow]) -> Path:
    """Plot fPC1/fPC2 positions for train, held-out and Lazar trials."""
    fig, axis = plt.subplots(figsize=(8, 6))
    for split_name, marker in SPLIT_MARKERS.items():
        for label in CLASS_LABELS:
            selected = [
                row
                for row in rows
                if row.split == split_name and row.true_label == label
            ]
            if not selected:
                continue
            scores = np.array([row.fpca_scores for row in selected])
            axis.scatter(
                scores[:, 0],
                scores[:, 1],
                marker=marker,
                s=75 if split_name != "train" else 42,
                color=CLASS_COLORS[label],
                edgecolor="black" if split_name != "train" else "none",
                linewidth=0.8,
                alpha=0.82,
                label=f"{split_name}: {label}",
            )

    for row in rows:
        if row.split == "train":
            continue
        axis.annotate(
            row.trial_name,
            (row.fpca_scores[0], row.fpca_scores[1]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )

    axis.set_title("fPCA score space: train vs held-out vs Lazar")
    axis.set_xlabel("fPC1 score")
    axis.set_ylabel("fPC2 score")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=7, ncols=2)
    return finish(fig, OUTPUT_DIR / "04_fpca_score_space_generalization.png")


def plot_prediction_table(rows: list[PredictionRow]) -> Path:
    """Save a compact correctness table for held-out and Lazar predictions."""
    visible_rows = [row for row in rows if row.split != "train"]
    fig, axis = plt.subplots(figsize=(11, 0.55 * len(visible_rows) + 1.2))
    axis.axis("off")
    table_data = [
        [
            row.split,
            row.trial_name,
            row.true_label,
            row.predicted_label,
            "OK" if row.true_label == row.predicted_label else "MISS",
        ]
        for row in visible_rows
    ]
    table = axis.table(
        cellText=table_data,
        colLabels=("Split", "Trial", "True", "Predicted", "Result"),
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#e8eef7")
        elif col_idx == 4:
            text = cell.get_text().get_text()
            cell.set_facecolor("#d8f3dc" if text == "OK" else "#ffd6d6")
    axis.set_title("Individual predictions not used for training", pad=12)
    return finish(fig, OUTPUT_DIR / "05_prediction_table.png")


def main() -> None:
    create_project_directories()
    train, heldout, lazar = split_trials()
    model = train_live_motion_model(train)

    train_rows = predict_trials(model, train, "train")
    heldout_rows = predict_trials(model, heldout, "anja_petar_test")
    lazar_rows = predict_trials(model, lazar, "lazar_external")
    all_rows = train_rows + heldout_rows + lazar_rows

    paths = [
        plot_split_counts(train, heldout, lazar),
        plot_confusion(
            heldout_rows,
            "Anja+Petar held-out test",
            "02_confusion_anja_petar_heldout.png",
        ),
        plot_confusion(
            lazar_rows,
            "Lazar external test",
            "03_confusion_lazar_external.png",
        ),
        plot_fpca_scores(all_rows),
        plot_prediction_table(all_rows),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
