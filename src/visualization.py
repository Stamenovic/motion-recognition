"""Plotting helpers for classification experiments."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .classification import ClassificationResult


def _style_axis(axis, y_format: str = "%.2f") -> None:
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15, linestyle=":")
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_major_formatter(FormatStrFormatter(y_format))


def _finish_figure(fig, output_path: Path | None, show: bool) -> Path | None:
    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_model_accuracy_comparison(
    results: dict[str, ClassificationResult],
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot accuracy for several SVM feature representations."""
    names = list(results)
    accuracies = [results[name].accuracy for name in names]

    fig, axis = plt.subplots(figsize=(9, 4.5))
    bars = axis.bar(names, accuracies, color=["tab:blue", "tab:green", "tab:purple"])
    axis.set_title("Standardized SVM model comparison")
    axis.set_ylabel("Leave-One-Out accuracy")
    axis.set_ylim(0.0, 1.05)
    _style_axis(axis)

    for bar, value in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    return _finish_figure(fig, output_path, show)


def plot_confusion_matrix(
    result: ClassificationResult,
    title: str,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot a confusion matrix for one classification result."""
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(result.confusion, cmap="Blues")
    fig.colorbar(image, ax=axis)

    axis.set_title(f"{title} | accuracy={result.accuracy:.3f}")
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

    return _finish_figure(fig, output_path, show)
