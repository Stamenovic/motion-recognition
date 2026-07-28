"""Plotting helpers for inspecting each processing step."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .classification import ClassificationResult
from .data_types import TrialRecord
from .feature_extraction import TrialFeatures
from .temporal_normalization import NormalizedTrialFeatures


def _style_axis(axis, y_format: str = "%.1f") -> None:
    axis.grid(True, which="major", alpha=0.35)
    axis.grid(True, which="minor", alpha=0.15, linestyle=":")
    axis.xaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_minor_locator(AutoMinorLocator())
    axis.yaxis.set_major_formatter(FormatStrFormatter(y_format))


def _mark_signal_max(axis, time, signal, color: str) -> None:
    max_idx = int(signal.argmax())
    axis.scatter(
        time[max_idx],
        signal[max_idx],
        color=color,
        s=28,
        zorder=3,
    )
    axis.annotate(
        f"max {signal[max_idx]:.1f}",
        xy=(time[max_idx], signal[max_idx]),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=8,
        color=color,
    )


def plot_raw_vs_filtered_translation(
    raw_trial: TrialRecord,
    filtered_trial: TrialRecord,
    segment_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot raw-vs-filtered translation for one segment."""
    raw_segment = raw_trial.segments[segment_name]
    filtered_segment = filtered_trial.segments[segment_name]

    time = [
        frame_idx * raw_trial.metadata.dt
        for frame_idx in range(raw_trial.metadata.num_frames)
    ]
    axes_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"{raw_trial.trial_name} | {segment_name} | raw vs filtered translation"
    )

    for axis_idx, axis in enumerate(axes):
        axis.plot(
            time,
            raw_segment.translation[:, axis_idx],
            label="raw",
            linewidth=1.0,
            alpha=0.65,
        )
        axis.plot(
            time,
            filtered_segment.translation[:, axis_idx],
            label="filtered",
            linewidth=1.4,
        )
        axis.set_ylabel(f"{axes_labels[axis_idx]} [mm]")
        _style_axis(axis)

    axes[-1].set_xlabel("Time [s]")
    axes[-1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[0].legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_raw_vs_filtered_translation_side_by_side(
    raw_trial: TrialRecord,
    filtered_trial: TrialRecord,
    segment_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot raw and filtered translations side by side for one segment."""
    raw_segment = raw_trial.segments[segment_name]
    filtered_segment = filtered_trial.segments[segment_name]

    time = [
        frame_idx * raw_trial.metadata.dt
        for frame_idx in range(raw_trial.metadata.num_frames)
    ]
    axes_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
    fig.suptitle(
        f"{raw_trial.trial_name} | {segment_name} | translation preprocessing"
    )

    for axis_idx, axis_label in enumerate(axes_labels):
        raw_axis = axes[axis_idx, 0]
        filtered_axis = axes[axis_idx, 1]

        raw_axis.plot(
            time,
            raw_segment.translation[:, axis_idx],
            color="tab:blue",
            linewidth=1.0,
        )
        filtered_axis.plot(
            time,
            filtered_segment.translation[:, axis_idx],
            color="tab:orange",
            linewidth=1.2,
        )

        raw_axis.set_ylabel(f"{axis_label} [mm]")
        _style_axis(raw_axis)
        _style_axis(filtered_axis)

        if axis_idx == 0:
            raw_axis.set_title("Raw")
            filtered_axis.set_title("Filtered")

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[-1, 0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[-1, 1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_trial_features(
    features: TrialFeatures,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot extracted distance, speed and acceleration signals for one trial."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"{features.trial_name} | extracted features | {features.label}")

    distance_names = [
        "distance_left_right",
        "distance_left_trunk",
        "distance_right_trunk",
    ]
    speed_names = ["speed_left", "speed_right"]
    acceleration_names = ["acceleration_left", "acceleration_right"]

    for name in distance_names:
        line = axes[0].plot(features.time, features.signals[name], label=name)[0]
        _mark_signal_max(axes[0], features.time, features.signals[name], line.get_color())
    axes[0].set_ylabel("Distance [mm]")
    _style_axis(axes[0])
    axes[0].legend(loc="best")

    for name in speed_names:
        line = axes[1].plot(features.time, features.signals[name], label=name)[0]
        _mark_signal_max(axes[1], features.time, features.signals[name], line.get_color())
    axes[1].set_ylabel("Speed [mm/s]")
    _style_axis(axes[1])
    axes[1].legend(loc="best")

    for name in acceleration_names:
        line = axes[2].plot(features.time, features.signals[name], label=name)[0]
        _mark_signal_max(axes[2], features.time, features.signals[name], line.get_color())
    axes[2].set_ylabel("Acceleration [mm/s^2]")
    axes[2].set_xlabel("Time [s]")
    axes[2].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _style_axis(axes[2])
    axes[2].legend(loc="best")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_original_vs_normalized_feature(
    features: TrialFeatures,
    normalized_features: NormalizedTrialFeatures,
    signal_name: str,
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot one feature before and after temporal normalization."""
    original_signal = features.signals[signal_name]
    normalized_signal = normalized_features.signals[signal_name]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(
        f"{features.trial_name} | {signal_name} | temporal normalization"
    )

    axes[0].plot(features.time, original_signal, color="tab:blue")
    axes[0].set_title(
        f"Original ({normalized_features.original_num_frames} frames, "
        f"{normalized_features.original_duration:.2f}s)"
    )
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Value")
    axes[0].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _style_axis(axes[0])
    _mark_signal_max(axes[0], features.time, original_signal, "tab:blue")

    axes[1].plot(
        normalized_features.normalized_time * 100.0,
        normalized_signal,
        color="tab:green",
    )
    axes[1].set_title(f"Normalized ({len(normalized_signal)} samples)")
    axes[1].set_xlabel("Movement duration [%]")
    axes[1].set_ylabel("Value")
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    _style_axis(axes[1])
    _mark_signal_max(
        axes[1],
        normalized_features.normalized_time * 100.0,
        normalized_signal,
        "tab:green",
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


def plot_model_accuracy_comparison(
    results: dict[str, ClassificationResult],
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot accuracy values for several classification models."""
    model_names = list(results)
    accuracies = [results[name].accuracy for name in model_names]

    fig, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(model_names, accuracies, color=["tab:blue", "tab:green", "tab:orange"])
    axis.set_title("SVM Leave-One-Out model comparison")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.05)
    _style_axis(axis, y_format="%.2f")

    for bar, accuracy in zip(bars, accuracies):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            accuracy,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
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


def plot_normalized_trial_comparison(
    normalized_trials: list[NormalizedTrialFeatures],
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot normalized distance/speed/acceleration features for several trials."""
    if not normalized_trials:
        raise ValueError("At least one normalized trial is required for comparison.")

    feature_groups = [
        (
            "Distances [mm]",
            [
                "distance_left_right",
                "distance_left_trunk",
                "distance_right_trunk",
            ],
        ),
        ("Speeds [mm/s]", ["speed_left", "speed_right"]),
        ("Accelerations [mm/s^2]", ["acceleration_left", "acceleration_right"]),
    ]

    fig, axes = plt.subplots(
        len(feature_groups),
        len(normalized_trials),
        figsize=(6 * len(normalized_trials), 10),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle("Normalized feature comparison")

    for col_idx, trial_features in enumerate(normalized_trials):
        x_axis = trial_features.normalized_time * 100.0
        axes[0, col_idx].set_title(
            f"{trial_features.trial_name}\nlabel={trial_features.label}"
        )

        for row_idx, (group_label, signal_names) in enumerate(feature_groups):
            axis = axes[row_idx, col_idx]
            for signal_name in signal_names:
                axis.plot(
                    x_axis,
                    trial_features.signals[signal_name],
                    label=signal_name,
                    linewidth=1.1,
                )

            axis.set_ylabel(group_label)
            axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            _style_axis(axis)
            if col_idx == len(normalized_trials) - 1:
                axis.legend(loc="best", fontsize=8)

    for axis in axes[-1, :]:
        axis.set_xlabel("Movement duration [%]")

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_scalar_feature_comparison(
    normalized_trials: list[NormalizedTrialFeatures],
    scalar_feature_names: list[str],
    output_path: Path | None = None,
    show: bool = True,
) -> Path | None:
    """Plot scalar trend features for several normalized trials."""
    if not normalized_trials:
        raise ValueError("At least one normalized trial is required for comparison.")

    trial_labels = [
        f"{item.trial_name}\n{item.label}"
        for item in normalized_trials
    ]
    x_positions = range(len(trial_labels))

    fig, axes = plt.subplots(
        len(scalar_feature_names),
        1,
        figsize=(10, 2.8 * len(scalar_feature_names)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle("Distance left-right trend feature comparison")

    for row_idx, feature_name in enumerate(scalar_feature_names):
        axis = axes[row_idx, 0]
        values = [
            item.scalar_features[feature_name]
            for item in normalized_trials
        ]
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
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
