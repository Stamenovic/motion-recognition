"""Replay a real CSV trial as a UDP stream for live-pipeline testing.

This is more useful than the synthetic sender for model validation because the
coordinates come from the same kind of Vicon export used during training.
"""
import argparse
import os
import socket
import struct
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import RAW_DATA_DIR
from src.data_loader import load_trials, load_vicon_csv
from src.data_types import TrialRecord
from src.feature_extraction import LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT
from src.path_parser import parse_trial_path


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 51001
DEFAULT_REST_SECONDS = 3.0

REPLAY_SEGMENTS = {
    "Left": LEFT_SEGMENT,
    "Right": RIGHT_SEGMENT,
    "Trup": TRUNK_SEGMENT,
}

_HEADER = struct.Struct("<IB")
_ITEM_HEADER = struct.Struct("<BH")
_ITEM_BODY = struct.Struct("<24s6d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Target address.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Target UDP port.")
    parser.add_argument("--trial", help="Trial name to replay, for example Sirenje_01.")
    parser.add_argument(
        "--trial-path",
        type=Path,
        help="Exact CSV path to replay when trial names are duplicated.",
    )
    parser.add_argument(
        "--list-trials",
        action="store_true",
        help="Print available labelled trials and exit.",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=DEFAULT_REST_SECONDS,
        help="Rest stream duration before and after the replayed movement.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="How many times to replay the trial. 0 means run until Ctrl+C.",
    )
    return parser.parse_args()


def encode_binary(frame_number: int, pose: dict[str, tuple[float, float, float]]) -> bytes:
    """Encode one frame in the Vicon UDP Object Stream binary layout."""
    packet = _HEADER.pack(frame_number, len(pose))
    for name, (tx, ty, tz) in pose.items():
        body = _ITEM_BODY.pack(name.encode("ascii"), tx, ty, tz, 0.0, 0.0, 0.0)
        packet += _ITEM_HEADER.pack(0, len(body)) + body
    return packet


def load_trial_from_path(csv_path: Path) -> TrialRecord:
    """Load one exact CSV path as a TrialRecord."""
    csv_path = csv_path.resolve()
    parsed = parse_trial_path(csv_path, RAW_DATA_DIR)
    if parsed is None:
        raise RuntimeError(f"Cannot infer movement label from {csv_path.name}.")

    metadata, segments = load_vicon_csv(csv_path)
    return TrialRecord(
        path=csv_path,
        patient=parsed["patient"],
        session=parsed["session"],
        trial_name=parsed["trial_name"],
        label=parsed["label"],
        metadata=metadata,
        segments=segments,
    )


def load_trial_by_name(trial_name: str):
    """Load the requested trial from data/raw."""
    trials = load_trials(RAW_DATA_DIR)
    if not trials:
        raise RuntimeError(f"No labelled CSV trials found under {RAW_DATA_DIR}.")

    if trial_name is None:
        available = ", ".join(sorted(trial.trial_name for trial in trials))
        raise RuntimeError(f"Pass --trial. Available trials: {available}")

    matches = [trial for trial in trials if trial.trial_name == trial_name]
    if not matches:
        available = ", ".join(sorted({trial.trial_name for trial in trials}))
        raise RuntimeError(f"Unknown trial {trial_name}. Available trials: {available}")
    if len(matches) > 1:
        paths = "\n".join(str(trial.path.relative_to(RAW_DATA_DIR)) for trial in matches)
        raise RuntimeError(
            f"Trial name {trial_name} is duplicated. Use --trial-path with one of:\n{paths}"
        )
    return matches[0]


def print_trials() -> None:
    """Print available trials grouped by label."""
    trials = load_trials(RAW_DATA_DIR)
    if not trials:
        print(f"No labelled CSV trials found under {RAW_DATA_DIR}.")
        return

    for trial in sorted(trials, key=lambda item: (item.label, item.trial_name)):
        path = trial.path.relative_to(RAW_DATA_DIR)
        print(f"{path}\t{trial.label}\t{trial.metadata.fps} fps")


def pose_from_trial_frame(trial, frame_index: int) -> dict[str, tuple[float, float, float]]:
    """Extract the replay object pose for one trial frame."""
    pose = {}
    for stream_name, segment_name in REPLAY_SEGMENTS.items():
        if segment_name not in trial.segments:
            raise KeyError(f"Trial {trial.trial_name} is missing segment {segment_name}.")
        pose[stream_name] = tuple(trial.segments[segment_name].translation[frame_index])
    return pose


def send_pose(sock: socket.socket, target, frame_number: int, pose: dict) -> None:
    """Send one encoded pose."""
    sock.sendto(encode_binary(frame_number, pose), target)


def sleep_until(deadline: float) -> None:
    """Sleep until the deadline if the sender is ahead of schedule."""
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def main() -> None:
    args = parse_args()
    if args.list_trials:
        print_trials()
        return

    trial = load_trial_from_path(args.trial_path) if args.trial_path else load_trial_by_name(args.trial)
    fps = trial.metadata.fps
    target = (args.host, args.port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rest_frames = max(int(args.rest_seconds * fps), 1)
    rest_pose = pose_from_trial_frame(trial, 0)
    frame_number = 0
    cycle = 0
    next_send = time.perf_counter()

    print(f"Replaying {trial.trial_name} ({trial.label}) to {args.host}:{args.port}.")
    print(f"Frames: {trial.metadata.num_frames}, fps: {fps}")
    print("Press Ctrl+C to stop.")
    print()

    try:
        while True:
            cycle += 1
            if args.cycles and cycle > args.cycles:
                break

            print(f"[{cycle:03d}] rest {args.rest_seconds:.1f}s")
            for _ in range(rest_frames):
                send_pose(sock, target, frame_number, rest_pose)
                frame_number += 1
                next_send += 1.0 / fps
                sleep_until(next_send)

            print(f"[{cycle:03d}] REPLAY {trial.trial_name} ({trial.label})")
            for frame_index in range(trial.metadata.num_frames):
                send_pose(sock, target, frame_number, pose_from_trial_frame(trial, frame_index))
                frame_number += 1
                next_send += 1.0 / fps
                sleep_until(next_send)

            print(f"[{cycle:03d}] done")
            print()
            for _ in range(rest_frames):
                send_pose(sock, target, frame_number, rest_pose)
                frame_number += 1
                next_send += 1.0 / fps
                sleep_until(next_send)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"Stopped after {frame_number} frames.")


if __name__ == "__main__":
    main()
