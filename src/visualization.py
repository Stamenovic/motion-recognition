"""Plotting helpers for inspecting each processing step."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".matplotlib-cache"),
)

import matplotlib.pyplot as plt

from .data_types import TrialRecord


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

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
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
        axis.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")
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

    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
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
        raw_axis.grid(True, alpha=0.25)
        filtered_axis.grid(True, alpha=0.25)

        if axis_idx == 0:
            raw_axis.set_title("Raw")
            filtered_axis.set_title("Filtered")

    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
