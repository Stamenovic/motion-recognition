"""Plotting helpers for classification experiments."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .classification import ClassificationResult
from .data_types import TrialRecord
from .feature_extraction import extract_trial_features
from .fpca_features import FPCADataset
from .preprocessing import filter_trial_translations


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


def plot_filter_cutoff_translation_comparison(
    trials: list[TrialRecord],
    segment_name: str,
    cutoff_hz_values: tuple[float, ...] = (6.0, 10.0, 15.0, 20.0),
    order: int = 2,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Compare raw and low-pass filtered translations for selected trials."""
    if not trials:
        raise ValueError("At least one trial is required for cutoff comparison.")

    axes_labels = ("X", "Y", "Z")
    fig, axes = plt.subplots(
        nrows=len(axes_labels),
        ncols=len(trials),
        figsize=(7 * len(trials), 8),
        sharex=False,
        squeeze=False,
    )
    fig.suptitle(
        "Butterworth cutoff comparison "
        f"(zero-phase order={order}, effective order={order * 2})"
    )

    for col_idx, trial in enumerate(trials):
        if segment_name not in trial.segments:
            raise KeyError(f"Trial {trial.trial_name} is missing {segment_name}.")

        time = np.arange(trial.metadata.num_frames) * trial.metadata.dt
        raw_translation = trial.segments[segment_name].translation
        filtered_by_cutoff = {
            cutoff_hz: filter_trial_translations(
                trial,
                cutoff_hz=cutoff_hz,
                order=order,
            ).segments[segment_name].translation
            for cutoff_hz in cutoff_hz_values
        }

        for row_idx, axis_label in enumerate(axes_labels):
            axis = axes[row_idx, col_idx]
            axis.plot(
                time,
                raw_translation[:, row_idx],
                color="black",
                linewidth=1.4,
                alpha=0.75,
                label="Original",
            )
            for cutoff_hz, filtered_translation in filtered_by_cutoff.items():
                axis.plot(
                    time,
                    filtered_translation[:, row_idx],
                    linewidth=1.0,
                    label=f"{cutoff_hz:g} Hz",
                )

            if row_idx == 0:
                axis.set_title(f"{trial.label.capitalize()} ({trial.trial_name})")
            if col_idx == 0:
                axis.set_ylabel(f"{axis_label} translation [mm]")
            if row_idx == len(axes_labels) - 1:
                axis.set_xlabel("Time [s]")
            _style_axis(axis, y_format="%.1f")

    axes[0, -1].legend(loc="best", fontsize=8)
    return _finish_figure(fig, output_path, show)


def plot_filter_cutoff_feature_comparison(
    trials: list[TrialRecord],
    signal_name: str = "acceleration_left",
    cutoff_hz_values: tuple[float, ...] = (6.0, 10.0, 15.0, 20.0),
    order: int = 2,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Compare raw and filtered extracted feature signals for selected trials."""
    if not trials:
        raise ValueError("At least one trial is required for cutoff comparison.")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(trials),
        figsize=(7 * len(trials), 4.8),
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        f"{signal_name} cutoff comparison "
        f"(zero-phase order={order}, effective order={order * 2})"
    )

    for axis, trial in zip(axes[0], trials):
        raw_features = extract_trial_features(trial)
        axis.plot(
            raw_features.time,
            raw_features.signals[signal_name],
            color="black",
            linewidth=1.3,
            alpha=0.8,
            label="Original",
        )

        for cutoff_hz in cutoff_hz_values:
            filtered_trial = filter_trial_translations(
                trial,
                cutoff_hz=cutoff_hz,
                order=order,
            )
            filtered_features = extract_trial_features(filtered_trial)
            axis.plot(
                filtered_features.time,
                filtered_features.signals[signal_name],
                linewidth=1.1,
                label=f"{cutoff_hz:g} Hz",
            )

        axis.set_title(f"{trial.label.capitalize()} ({trial.trial_name})")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(signal_name)
        _style_axis(axis, y_format="%.1f")

    axes[0, -1].legend(loc="best", fontsize=8)
    return _finish_figure(fig, output_path, show)


def plot_filter_cutoff_feature_stacked(
    trials: list[TrialRecord],
    signal_name: str = "acceleration_left",
    cutoff_hz_values: tuple[float, ...] = (6.0, 10.0, 15.0, 20.0),
    order: int = 2,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Compare all cutoff values in one column with one trial per row."""
    if not trials:
        raise ValueError("At least one trial is required for cutoff comparison.")

    fig, axes = plt.subplots(
        nrows=len(trials),
        ncols=1,
        figsize=(12, 4.2 * len(trials)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        f"{signal_name} cutoff comparison "
        f"(zero-phase order={order}, effective order={order * 2})"
    )

    for axis, trial in zip(axes[:, 0], trials):
        raw_features = extract_trial_features(trial)
        axis.plot(
            raw_features.time,
            raw_features.signals[signal_name],
            color="black",
            linewidth=1.3,
            alpha=0.8,
            label="Original",
        )

        for cutoff_hz in cutoff_hz_values:
            filtered_trial = filter_trial_translations(
                trial,
                cutoff_hz=cutoff_hz,
                order=order,
            )
            filtered_features = extract_trial_features(filtered_trial)
            axis.plot(
                filtered_features.time,
                filtered_features.signals[signal_name],
                linewidth=1.1,
                label=f"{cutoff_hz:g} Hz",
            )

        axis.set_title(f"{trial.label.capitalize()} ({trial.trial_name})")
        axis.set_xlabel("Time [s]")
        axis.set_ylabel(signal_name)
        axis.legend(loc="best", fontsize=8)
        _style_axis(axis, y_format="%.1f")

    return _finish_figure(fig, output_path, show)


def plot_filter_cutoff_feature_grid(
    trials: list[TrialRecord],
    signal_name: str = "acceleration_left",
    cutoff_hz_values: tuple[float, ...] = (6.0, 10.0, 15.0, 20.0),
    order: int = 2,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Compare one cutoff per row against the original feature signal."""
    if not trials:
        raise ValueError("At least one trial is required for cutoff comparison.")

    fig, axes = plt.subplots(
        nrows=len(cutoff_hz_values),
        ncols=len(trials),
        figsize=(7 * len(trials), 3.1 * len(cutoff_hz_values)),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    fig.suptitle(
        f"{signal_name}: original vs one Butterworth cutoff per row "
        f"(zero-phase order={order}, effective order={order * 2})"
    )

    for col_idx, trial in enumerate(trials):
        raw_features = extract_trial_features(trial)
        raw_signal = raw_features.signals[signal_name]

        for row_idx, cutoff_hz in enumerate(cutoff_hz_values):
            axis = axes[row_idx, col_idx]
            filtered_trial = filter_trial_translations(
                trial,
                cutoff_hz=cutoff_hz,
                order=order,
            )
            filtered_features = extract_trial_features(filtered_trial)

            axis.plot(
                raw_features.time,
                raw_signal,
                color="black",
                linewidth=1.0,
                alpha=0.65,
                label="Original",
            )
            axis.plot(
                filtered_features.time,
                filtered_features.signals[signal_name],
                color="tab:red",
                linewidth=1.2,
                label=f"{cutoff_hz:g} Hz",
            )

            if row_idx == 0:
                axis.set_title(f"{trial.label.capitalize()} ({trial.trial_name})")
            if col_idx == 0:
                axis.set_ylabel(f"{cutoff_hz:g} Hz\n{signal_name}")
            if row_idx == len(cutoff_hz_values) - 1:
                axis.set_xlabel("Time [s]")
            if row_idx == 0 and col_idx == len(trials) - 1:
                axis.legend(loc="best", fontsize=8)
            _style_axis(axis, y_format="%.1f")

    return _finish_figure(fig, output_path, show)


def plot_winter_residual_translation_analysis(
    trials: list[TrialRecord],
    segment_name: str,
    cutoff_hz_values: tuple[float, ...] = tuple(np.arange(2.0, 45.0, 1.0)),
    order: int = 2,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot Winter-style residual curves over raw XYZ translations."""
    if not trials:
        raise ValueError("At least one trial is required for residual analysis.")

    fig, axis = plt.subplots(figsize=(8, 5))
    for trial in trials:
        if segment_name not in trial.segments:
            raise KeyError(f"Trial {trial.trial_name} is missing {segment_name}.")

        raw_translation = trial.segments[segment_name].translation
        valid_cutoffs = [
            cutoff_hz
            for cutoff_hz in cutoff_hz_values
            if cutoff_hz < trial.metadata.fps / 2.0
        ]
        residuals = []
        for cutoff_hz in valid_cutoffs:
            filtered_translation = filter_trial_translations(
                trial,
                cutoff_hz=cutoff_hz,
                order=order,
            ).segments[segment_name].translation
            residuals.append(
                float(np.sqrt(np.mean((raw_translation - filtered_translation) ** 2)))
            )

        axis.plot(
            valid_cutoffs,
            residuals,
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"{trial.label.capitalize()} ({trial.trial_name})",
        )

    axis.set_title(f"Winter residual analysis approximation | {segment_name}")
    axis.set_xlabel("Cutoff frequency [Hz]")
    axis.set_ylabel("XYZ RMS residual [mm]")
    axis.legend(loc="best")
    _style_axis(axis, y_format="%.2f")
    return _finish_figure(fig, output_path, show)


def plot_fpca_score_scatter(
    dataset: FPCADataset,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot fPC1 vs fPC2 scores, colored by movement label."""
    if dataset.X.shape[1] < 2:
        raise ValueError("At least two fPCA components are required for scatter plot.")

    fig, axis = plt.subplots(figsize=(7, 5.5))
    colors = {"guranje": "tab:blue", "sirenje": "tab:orange"}
    markers = {"guranje": "o", "sirenje": "s"}

    for label in sorted(set(dataset.y)):
        mask = dataset.y == label
        axis.scatter(
            dataset.X[mask, 0],
            dataset.X[mask, 1],
            label=str(label).capitalize(),
            s=80,
            color=colors.get(str(label)),
            marker=markers.get(str(label), "o"),
            alpha=0.85,
        )

    for trial_name, x_value, y_value in zip(
        dataset.trial_names,
        dataset.X[:, 0],
        dataset.X[:, 1],
    ):
        axis.annotate(
            trial_name,
            (x_value, y_value),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    variance = dataset.explained_variance_ratio
    fpc1_label = "fPC1 score"
    fpc2_label = "fPC2 score"
    if len(variance) >= 2:
        fpc1_label = f"fPC1 ({variance[0] * 100:.1f}% variance)"
        fpc2_label = f"fPC2 ({variance[1] * 100:.1f}% variance)"

    axis.set_title("fPCA score scatter")
    axis.set_xlabel(fpc1_label)
    axis.set_ylabel(fpc2_label)
    axis.legend(loc="best")
    _style_axis(axis, y_format="%.2f")
    return _finish_figure(fig, output_path, show)
