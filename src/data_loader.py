"""Load Vicon Nexus CSV trial exports and discover movement trials."""
import csv
from pathlib import Path

import numpy as np

from .data_types import SegmentTrajectory, TrialMetadata, TrialRecord
from .path_parser import parse_trial_path


def load_vicon_csv(csv_path: Path) -> tuple[TrialMetadata, dict[str, SegmentTrajectory]]:
    """Parse the Segments section of a Vicon Nexus CSV export."""
    with open(csv_path, "r", newline="") as f:
        all_rows = list(csv.reader(f))

    start_idx = next(
        (i for i, row in enumerate(all_rows) if row and row[0].strip() == "Segments"),
        None,
    )
    if start_idx is None:
        raise ValueError(f"No 'Segments' marker found in {csv_path}.")

    fps = int(all_rows[start_idx + 1][0].strip())
    name_row = all_rows[start_idx + 2]
    segments_meta = [
        (name.strip(), col_idx)
        for col_idx, name in enumerate(name_row)
        if col_idx >= 2 and name.strip()
    ]

    numeric_rows = []
    for row in all_rows[start_idx + 5 :]:
        if not row or not row[0].strip():
            break
        numeric_rows.append([float(v) if v.strip() else np.nan for v in row])

    data = np.array(numeric_rows)
    if data.size == 0:
        raise ValueError(f"No segment frame data found in {csv_path}.")

    frames = data[:, 0]
    metadata = TrialMetadata(
        fps=fps,
        dt=1.0 / fps,
        num_frames=len(frames),
        frame_start=int(frames[0]),
        frame_end=int(frames[-1]),
    )
    segments = {
        name: SegmentTrajectory(
            name=name,
            rotation=data[:, col : col + 3],
            translation=data[:, col + 3 : col + 6],
        )
        for name, col in segments_meta
    }
    return metadata, segments


def iter_csv_paths(root: Path) -> list[Path]:
    """Find real CSV exports recursively, skipping office/Vicon lock files."""
    return sorted(
        path
        for path in root.rglob("*.csv")
        if path.is_file() and not path.name.startswith(".~lock.")
    )


def load_trials(root: Path) -> list[TrialRecord]:
    """Discover and load labeled Vicon CSV trials under the root folder."""
    trials: list[TrialRecord] = []
    for csv_path in iter_csv_paths(root):
        parsed = parse_trial_path(csv_path, root)
        if parsed is None:
            continue

        metadata, segments = load_vicon_csv(csv_path)
        trials.append(
            TrialRecord(
                path=csv_path,
                patient=parsed["patient"],
                session=parsed["session"],
                trial_name=parsed["trial_name"],
                label=parsed["label"],
                metadata=metadata,
                segments=segments,
            )
        )
    return trials
