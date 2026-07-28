"""Plotting helpers for inspecting the no-normalization SVM experiment."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .classification import ClassificationResult
from .statistical_features import StatisticalTrialFeatures


def _style_axis(axis, y_format: str = "%.2f") -> None:
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15, linestyle=":")
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_major_formatter(FormatStrFormatter(y_format))


def plot_confusion_matrix(
    result: ClassificationResult,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot a confusion matrix for a classification result."""
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(result.confusion, cmap="Blues")
    fig.colorbar(image, ax=axis)

    axis.set_title(f"SVM Leave-One-Out | accuracy={result.accuracy:.3f}")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(result.labels)), result.labels)
    axis.set_yticks(range(len(result.labels)), result.labels)

    max_value = result.confusion.max() if result.confusion.size else 0
    for row_idx in range(result.confusion.shape[0]):
        for col_idx in range(result.confusion.shape[1]):
            value = result.confusion[row_idx, col_idx]
            text_color = "white" if value > max_value / 2 else "black"
            axis.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_accuracy_bar(
    result: ClassificationResult,
    title: str,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot one accuracy bar for the no-normalization experiment."""
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(["statistical features"], [result.accuracy], color="tab:purple")
    axis.set_title(title)
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.05)
    _style_axis(axis)
    axis.text(0, result.accuracy, f"{result.accuracy:.3f}", ha="center", va="bottom")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_statistical_feature_comparison(
    statistical_trials: list[StatisticalTrialFeatures],
    feature_names: list[str],
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Compare selected scalar statistical features across several trials."""
    if not statistical_trials:
        raise ValueError("At least one trial is required for comparison.")

    trial_labels = [f"{item.trial_name}\n{item.label}" for item in statistical_trials]
    x_positions = range(len(trial_labels))

    fig, axes = plt.subplots(
        len(feature_names),
        1,
        figsize=(10, 2.7 * len(feature_names)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle("No-normalization statistical feature comparison")

    for row_idx, feature_name in enumerate(feature_names):
        axis = axes[row_idx, 0]
        values = [item.values[feature_name] for item in statistical_trials]
        bars = axis.bar(x_positions, values, color=["tab:blue", "tab:green", "tab:red"])
        axis.set_ylabel(feature_name.replace("distance_left_right_", ""))
        _style_axis(axis)

        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axes[-1, 0].set_xticks(list(x_positions), trial_labels)
    explanation = (
        "delta = end - start distance\n"
        "range = max - min distance\n"
        "positive_slope_ratio = fraction of intervals where distance increases\n"
        "time_to_max_ratio = time of maximum distance / total movement duration"
    )
    fig.text(0.02, 0.015, explanation, ha="left", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
