"""Test rolling-window live predictions over several concatenated CSV trials."""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import RAW_DATA_DIR
from src.data_loader import load_trials
from src.feature_extraction import LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT
from src.live_capture import LiveFrame, LiveSegmentBuffer
from src.live_model import train_live_motion_model


REQUIRED_SEGMENTS = (LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT)
DEFAULT_TRIAL_PATHS = (
    RAW_DATA_DIR / "data2026" / "Petar" / "Snimanje" / "Guranje01.csv",
    RAW_DATA_DIR / "data2026" / "Petar" / "Snimanje" / "Sirenje01.csv",
    RAW_DATA_DIR / "data2026" / "Petar" / "Snimanje" / "Podizanje_desna01.csv",
    RAW_DATA_DIR / "data2026" / "Lazar (asistent)" / "Snimanje" / "Sirenje_03.csv",
    RAW_DATA_DIR / "data2026" / "Lazar (asistent)" / "Snimanje" / "Guranje_01.csv",
    RAW_DATA_DIR / "data2026" / "Anja" / "Sirenje" / "Guranje01.csv",
    RAW_DATA_DIR / "data2026" / "Anja" / "Sirenje" / "Sirenje01.csv",
    RAW_DATA_DIR / "data2026" / "Anja" / "Sirenje" / "Podizanje_desna01.csv",
    RAW_DATA_DIR / "data2026" / "Lazar (asistent)" / "Snimanje" / "Podizanje_desna_01.csv",
    RAW_DATA_DIR / "data2026" / "Petar" / "Snimanje" / "Guranje03.csv",
    RAW_DATA_DIR / "data2026" / "Petar" / "Snimanje" / "Sirenje04.csv",
    RAW_DATA_DIR / "data2026" / "Anja" / "Sirenje" / "Guranje05.csv",
    RAW_DATA_DIR / "data2026" / "Lazar (asistent)" / "Snimanje" / "Podizanje_desna_02.csv",
    RAW_DATA_DIR / "data2026" / "Anja" / "Sirenje" / "Sirenje05.csv",
    RAW_DATA_DIR / "data2026" / "Petar" / "Snimanje" / "Podizanje_desna03.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--window-frames",
        type=int,
        default=1000,
        help="Rolling window length.",
    )
    parser.add_argument(
        "--prediction-stride",
        type=int,
        default=10,
        help="Predict every N complete frames.",
    )
    parser.add_argument(
        "--rest-frames",
        type=int,
        default=100,
        help="Number of repeated rest-pose frames between trials.",
    )
    parser.add_argument(
        "--long-rest-after",
        type=int,
        default=5,
        help="Insert a longer rest block after this many movements. 0 disables it.",
    )
    parser.add_argument(
        "--long-rest-frames",
        type=int,
        default=600,
        help="Number of repeated rest-pose frames in the inserted long rest block.",
    )
    parser.add_argument(
        "--initial-rest-frames",
        type=int,
        default=0,
        help="Number of repeated rest-pose frames before the first movement.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=300,
        help="Do not predict before the buffer has at least this many frames. "
        "Defaults to the live server value.",
    )
    parser.add_argument(
        "--stable-predictions",
        type=int,
        default=5,
        help="Emit an event after this many equal known predictions in a row.",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=300,
        help="Frames to ignore after an emitted event.",
    )
    parser.add_argument(
        "--trial-path",
        type=Path,
        action="append",
        help="Exact CSV path to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--trigger-test",
        action="store_true",
        help="Also test automatic start/stop triggering from a shared rest pose.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=50,
        help="Rest frames used to estimate the starting pose baseline.",
    )
    parser.add_argument(
        "--start-delta-mm",
        type=float,
        default=80.0,
        help="Start recording when hand displacement from baseline exceeds this value.",
    )
    parser.add_argument(
        "--start-speed-mm-s",
        type=float,
        default=150.0,
        help="Start recording when hand speed exceeds this value.",
    )
    parser.add_argument(
        "--stop-speed-mm-s",
        type=float,
        default=120.0,
        help="Count a frame as quiet when hand speed is below this value.",
    )
    parser.add_argument(
        "--stop-quiet-frames",
        type=int,
        default=40,
        help="Stop recording after this many quiet frames.",
    )
    parser.add_argument(
        "--max-segment-frames",
        type=int,
        default=1000,
        help="Force classification when a triggered segment reaches this length.",
    )
    parser.add_argument(
        "--include-baseline-in-segment",
        action="store_true",
        help="Include baseline frames before the trigger in the classified segment.",
    )
    return parser.parse_args()


def _load_exact_trials(paths: tuple[Path, ...] | list[Path]):
    all_trials = load_trials(RAW_DATA_DIR)
    by_path = {trial.path.resolve(): trial for trial in all_trials}
    selected = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in by_path:
            raise RuntimeError(f"Trial was not loaded or is not labelled: {path}")
        selected.append(by_path[resolved])
    return all_trials, selected


def _frame_from_trial(trial, frame_index: int, frame_number: int) -> LiveFrame:
    translations = {
        segment_name: tuple(trial.segments[segment_name].translation[frame_index])
        for segment_name in REQUIRED_SEGMENTS
    }
    rotations = {
        segment_name: tuple(trial.segments[segment_name].rotation[frame_index])
        for segment_name in REQUIRED_SEGMENTS
    }
    return LiveFrame(
        frame_number=frame_number,
        timestamp_seconds=frame_number / trial.metadata.fps,
        translations=translations,
        rotations=rotations,
    )


def _append_rest_frames(frames, trial, frame_number: int, count: int) -> int:
    for _ in range(count):
        frames.append(_frame_from_trial(trial, 0, frame_number))
        frame_number += 1
    return frame_number


def build_continuous_frames(
    trials,
    rest_frames: int,
    initial_rest_frames: int,
    long_rest_after: int,
    long_rest_frames: int,
) -> tuple[list[LiveFrame], list[dict]]:
    """Concatenate trials into one stream and keep expected-label intervals."""
    frames = []
    intervals = []
    frame_number = 0

    if trials and initial_rest_frames:
        rest_start = frame_number
        frame_number = _append_rest_frames(
            frames,
            trials[0],
            frame_number,
            initial_rest_frames,
        )
        intervals.append(
            {
                "trial": "initial_rest",
                "label": "rest/mixed",
                "start": rest_start,
                "end": frame_number - 1,
            }
        )

    for index, trial in enumerate(trials, start=1):
        frame_number = _append_rest_frames(frames, trial, frame_number, rest_frames)
        movement_start = frame_number
        for frame_index in range(trial.metadata.num_frames):
            frames.append(_frame_from_trial(trial, frame_index, frame_number))
            frame_number += 1
        movement_end = frame_number - 1
        intervals.append(
            {
                "trial": trial.trial_name,
                "label": trial.label,
                "start": movement_start,
                "end": movement_end,
            }
        )
        if long_rest_after and index == long_rest_after:
            rest_start = frame_number
            frame_number = _append_rest_frames(
                frames,
                trial,
                frame_number,
                long_rest_frames,
            )
            intervals.append(
                {
                    "trial": f"long_rest_after_{trial.trial_name}",
                    "label": "rest/mixed",
                    "start": rest_start,
                    "end": frame_number - 1,
                }
            )
    return frames, intervals


def expected_label_for_window(window_start: int, window_end: int, intervals) -> str:
    """Return the movement interval with the largest overlap with this window."""
    interval = expected_interval_for_window(window_start, window_end, intervals)
    return interval["label"] if interval else "rest/mixed"


def expected_interval_for_window(window_start: int, window_end: int, intervals) -> dict | None:
    """Return the movement interval with the largest overlap with this window."""
    best_interval = None
    best_overlap = 0
    for interval in intervals:
        overlap = max(
            0,
            min(window_end, interval["end"]) - max(window_start, interval["start"]) + 1,
        )
        if overlap > best_overlap:
            best_interval = interval
            best_overlap = overlap
    return best_interval


def _translation(frame: LiveFrame, segment_name: str) -> np.ndarray:
    return np.asarray(frame.translations[segment_name], dtype=float)


def _hand_motion_from_baseline(frame: LiveFrame, baseline: dict[str, np.ndarray]) -> float:
    left_delta = np.linalg.norm(_translation(frame, LEFT_SEGMENT) - baseline[LEFT_SEGMENT])
    right_delta = np.linalg.norm(_translation(frame, RIGHT_SEGMENT) - baseline[RIGHT_SEGMENT])
    return float(max(left_delta, right_delta))


def _max_hand_speed(previous: LiveFrame | None, frame: LiveFrame, fps: int) -> float:
    if previous is None:
        return 0.0
    left_speed = np.linalg.norm(
        _translation(frame, LEFT_SEGMENT) - _translation(previous, LEFT_SEGMENT)
    ) * fps
    right_speed = np.linalg.norm(
        _translation(frame, RIGHT_SEGMENT) - _translation(previous, RIGHT_SEGMENT)
    ) * fps
    return float(max(left_speed, right_speed))


def classify_triggered_segments(
    frames: list[LiveFrame],
    intervals,
    model,
    fps: int,
    args: argparse.Namespace,
) -> list[dict]:
    """Detect movement start/stop from rest pose and classify complete segments."""
    baseline_frames: list[LiveFrame] = []
    segment_buffer = LiveSegmentBuffer(fps=fps, required_segments=REQUIRED_SEGMENTS)
    events = []
    state = "baseline"
    baseline = None
    previous_frame = None
    quiet_count = 0
    cooldown_remaining = 0

    for frame in frames:
        speed = _max_hand_speed(previous_frame, frame, fps)
        previous_frame = frame

        if cooldown_remaining:
            cooldown_remaining -= 1
            continue

        if state == "baseline":
            baseline_frames.append(frame)
            if len(baseline_frames) > args.baseline_frames:
                baseline_frames = baseline_frames[-args.baseline_frames:]
            if len(baseline_frames) < args.baseline_frames:
                continue
            baseline = {
                segment_name: np.mean(
                    [_translation(item, segment_name) for item in baseline_frames],
                    axis=0,
                )
                for segment_name in REQUIRED_SEGMENTS
            }
            hand_delta = _hand_motion_from_baseline(frame, baseline)
            if hand_delta >= args.start_delta_mm or speed >= args.start_speed_mm_s:
                state = "recording"
                segment_buffer.clear()
                if args.include_baseline_in_segment:
                    for item in baseline_frames:
                        segment_buffer.append(item)
                segment_buffer.append(frame)
                quiet_count = 0
            continue

        segment_buffer.append(frame)
        if speed <= args.stop_speed_mm_s:
            quiet_count += 1
        else:
            quiet_count = 0

        long_enough = len(segment_buffer.frames) >= args.min_frames
        quiet_stop = long_enough and quiet_count >= args.stop_quiet_frames
        forced_stop = len(segment_buffer.frames) >= args.max_segment_frames
        if not quiet_stop and not forced_stop:
            continue

        trial = segment_buffer.to_trial_record(
            trial_name=f"triggered_segment_{len(events) + 1:03d}"
        )
        result = model.predict_trial(trial)
        interval = expected_interval_for_window(
            trial.metadata.frame_start,
            trial.metadata.frame_end,
            intervals,
        )
        events.append(
            {
                "frame": trial.metadata.frame_end,
                "range": f"{trial.metadata.frame_start}-{trial.metadata.frame_end}",
                "expected": interval["label"] if interval else "rest/mixed",
                "expected_trial": interval["trial"] if interval else "",
                "predicted": result.fpca_prediction,
                "candidate": result.known_prediction,
                "confidence": result.fpca_confidence,
                "motion": result.motion_extent_mm,
                "frames": trial.metadata.num_frames,
                "reason": "quiet" if quiet_stop else "max",
            }
        )
        state = "baseline"
        baseline_frames = []
        segment_buffer.clear()
        quiet_count = 0
        cooldown_remaining = args.cooldown_frames
    return events


def print_events(title: str, events: list[dict]) -> None:
    print(title)
    for index, event in enumerate(events, start=1):
        status = "OK" if event["expected"] == event["predicted"] else "MISS"
        extra = f", reason={event['reason']}" if "reason" in event else ""
        frame_range = f", range={event['range']}" if "range" in event else ""
        print(
            f"  {index}. frame={event['frame']}{frame_range}, "
            f"expected={event['expected']} {event['expected_trial']}, "
            f"predicted={event['predicted']}, "
            f"confidence={event['confidence']:.3f}, "
            f"motion={event['motion']:.1f}{extra} [{status}]"
        )


def main() -> None:
    args = parse_args()
    paths = args.trial_path if args.trial_path else DEFAULT_TRIAL_PATHS
    all_trials, selected_trials = _load_exact_trials(paths)
    model = train_live_motion_model(all_trials)
    window_frames = args.window_frames
    min_frames = args.min_frames or window_frames

    frames, intervals = build_continuous_frames(
        selected_trials,
        args.rest_frames,
        args.initial_rest_frames,
        args.long_rest_after,
        args.long_rest_frames,
    )
    buffer = LiveSegmentBuffer(
        fps=selected_trials[0].metadata.fps,
        required_segments=REQUIRED_SEGMENTS,
    )
    predictions = []
    events = []
    stable_label = None
    stable_count = 0
    cooldown_until = -1

    for frame_index, frame in enumerate(frames, start=1):
        if frame.frame_number < cooldown_until:
            continue
        buffer.append(frame)
        if len(buffer.frames) > window_frames:
            buffer.frames = buffer.frames[-window_frames:]
        if frame_index % args.prediction_stride:
            continue
        if len(buffer.frames) < min_frames:
            continue
        trial = buffer.to_trial_record(trial_name=f"rolling_window_{frame_index:06d}")
        result = model.predict_trial(trial)
        expected = expected_label_for_window(
            trial.metadata.frame_start,
            trial.metadata.frame_end,
            intervals,
        )
        predictions.append((trial.metadata.frame_end, expected, result.fpca_prediction))
        if result.is_unknown:
            stable_label = None
            stable_count = 0
            continue
        if result.fpca_prediction == stable_label:
            stable_count += 1
        else:
            stable_label = result.fpca_prediction
            stable_count = 1

        if stable_count >= args.stable_predictions:
            interval = expected_interval_for_window(
                trial.metadata.frame_start,
                trial.metadata.frame_end,
                intervals,
            )
            expected_event = interval["label"] if interval else "rest/mixed"
            expected_trial = interval["trial"] if interval else ""
            events.append(
                {
                    "frame": trial.metadata.frame_end,
                    "expected": expected_event,
                    "expected_trial": expected_trial,
                    "predicted": result.fpca_prediction,
                    "confidence": result.fpca_confidence,
                    "motion": result.motion_extent_mm,
                }
            )
            buffer.clear()
            stable_label = None
            stable_count = 0
            cooldown_until = trial.metadata.frame_end + args.cooldown_frames

    print(f"Selected trials: {len(selected_trials)}")
    for trial in selected_trials:
        print(f"  {trial.trial_name}: {trial.label}, {trial.metadata.num_frames} frames")
    print(f"Rolling window frames: {window_frames}")
    print(f"Prediction stride: {args.prediction_stride}")
    print(f"Minimum frames before prediction: {min_frames}")
    print(f"Stable predictions for event: {args.stable_predictions}")
    print(f"Cooldown frames after event: {args.cooldown_frames}")
    if args.long_rest_after:
        print(
            f"Inserted long rest: {args.long_rest_frames} frames "
            f"after movement {args.long_rest_after}"
        )
    if args.initial_rest_frames:
        print(f"Initial rest: {args.initial_rest_frames} frames")
    print(f"Total stream frames: {len(frames)}")
    print(f"Predictions: {len(predictions)}")
    print(f"Detected events: {len(events)}")
    print()

    print("Prediction counts by expected interval:")
    by_expected = {}
    for _, expected, prediction in predictions:
        by_expected.setdefault(expected, Counter())[prediction] += 1
    for expected, counts in sorted(by_expected.items()):
        print(f"  expected={expected}: {dict(sorted(counts.items()))}")

    print()
    print_events("Detected rolling events:", events)

    if args.trigger_test:
        trigger_events = classify_triggered_segments(
            frames,
            intervals,
            model,
            selected_trials[0].metadata.fps,
            args,
        )
        print()
        print("Trigger settings:")
        print(f"  baseline frames: {args.baseline_frames}")
        print(f"  start delta: {args.start_delta_mm:.1f} mm")
        print(f"  start speed: {args.start_speed_mm_s:.1f} mm/s")
        print(f"  stop speed: {args.stop_speed_mm_s:.1f} mm/s")
        print(f"  stop quiet frames: {args.stop_quiet_frames}")
        print(f"  max segment frames: {args.max_segment_frames}")
        print_events("Detected trigger events:", trigger_events)

    print()
    print("Last 30 predictions:")
    for frame_number, expected, prediction in predictions[-30:]:
        print(f"  frame={frame_number}, expected={expected}, predicted={prediction}")


if __name__ == "__main__":
    main()
