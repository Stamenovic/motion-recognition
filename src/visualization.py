"""Plotting helpers for inspecting each processing step."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .data_types import TrialRecord
from .feature_extraction import TrialFeatures


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
