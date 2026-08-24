"""UDP server that records Vicon Nexus frames and classifies movements.

Runtime flow:

1. listen for UDP packets streamed by Vicon Nexus,
2. parse each packet into per-object translations,
3. SPACE starts/stops a segment buffer,
4. convert the finished buffer to a TrialRecord,
5. classify it with the persisted live model.

Only this file talks to the network and the keyboard. Everything downstream
reuses the same modules the offline training pipeline uses.
"""
import argparse
import os
import socket
import struct
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import MODELS_DIR, RAW_DATA_DIR
from src.data_loader import load_trials
from src.feature_extraction import LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT
from src.live_capture import LiveFrame, LiveSegmentBuffer
from src.live_model import LiveMotionModel, train_live_motion_model

try:
    import msvcrt
except ImportError:  # pragma: no cover - this server is Windows-only
    msvcrt = None


MODEL_PATH = MODELS_DIR / "live_motion_model.joblib"
REQUIRED_SEGMENTS = (LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 51001
DEFAULT_FPS = 200  # must match the training CSVs in data/raw, which are 200 Hz
DEFAULT_MIN_FRAMES = 10

# Vicon streams bare object names; the trained model expects the CSV export
# names ("Subject:Segment"). Adjust the left side to match your Nexus objects.
OBJECT_NAME_MAP = {
    "Left": LEFT_SEGMENT,
    "Right": RIGHT_SEGMENT,
    "Trup": TRUNK_SEGMENT,
}

# Vicon UDP Object Stream binary layout (little endian).
_HEADER = struct.Struct("<IB")  # frame number, items in block
_ITEM_HEADER = struct.Struct("<BH")  # item id, item data size
_ITEM_BODY = struct.Struct("<24s6d")  # name, tx, ty, tz, rx, ry, rz
_ITEM_SIZE = _ITEM_HEADER.size + _ITEM_BODY.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Local address to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="UDP port to bind.")
    parser.add_argument(
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="Vicon capture rate. Must match the rate of the training CSV files.",
    )
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--use-saved-model",
        action="store_true",
        help="Load --model-path instead of training on all available CSV trials at startup.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=DEFAULT_MIN_FRAMES,
        help="Reject segments shorter than this many frames.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Print parsed packets and exit without loading the model. "
        "Use this first to confirm the stream format and object names.",
    )
    return parser.parse_args()


def load_or_train_model(args) -> LiveMotionModel:
    """Train on all available CSV trials for testing, or load a saved model."""
    if args.use_saved_model:
        return LiveMotionModel.load(args.model_path)

    trials = load_trials(RAW_DATA_DIR)
    if not trials:
        raise RuntimeError(f"No labeled CSV trials found under {RAW_DATA_DIR}.")

    print(f"Training test model on all available trials: {len(trials)}")
    print(f"Training labels: {dict(sorted(Counter(trial.label for trial in trials).items()))}")
    model = train_live_motion_model(trials)
    model.save(args.model_path)
    print(f"Saved refreshed model: {args.model_path}")
    return model


def parse_packet(data: bytes):
    """Parse one UDP packet into (frame_number, translations, rotations)."""
    parsed = _parse_binary_packet(data)
    if parsed is not None:
        return parsed
    return _parse_text_packet(data)


def _parse_binary_packet(data: bytes):
    """Parse the Vicon UDP Object Stream binary format, or return None."""
    if len(data) < _HEADER.size:
        return None
    payload_size = len(data) - _HEADER.size
    if payload_size == 0 or payload_size % _ITEM_SIZE != 0:
        return None

    frame_number, items_in_block = _HEADER.unpack_from(data, 0)
    if items_in_block != payload_size // _ITEM_SIZE:
        return None

    translations: dict[str, tuple[float, float, float]] = {}
    rotations: dict[str, tuple[float, float, float]] = {}
    offset = _HEADER.size
    for _ in range(items_in_block):
        offset += _ITEM_HEADER.size
        raw_name, tx, ty, tz, rx, ry, rz = _ITEM_BODY.unpack_from(data, offset)
        offset += _ITEM_BODY.size
        name = raw_name.split(b"\x00", 1)[0].decode("ascii", errors="ignore").strip()
        if not name:
            continue
        translations[name] = (tx, ty, tz)
        rotations[name] = (rx, ry, rz)
    return frame_number, translations, rotations


def _parse_text_packet(data: bytes):
    """Parse a plain-text fallback packet.

    Accepted lines:
        frame=<number>
        <object_name>,<tx>,<ty>,<tz>
        <object_name>,<tx>,<ty>,<tz>,<rx>,<ry>,<rz>
    """
    translations: dict[str, tuple[float, float, float]] = {}
    rotations: dict[str, tuple[float, float, float]] = {}
    frame_number = None

    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return frame_number, translations, rotations

    for raw_line in text.replace(";", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower().startswith("frame="):
            try:
                frame_number = int(float(line.split("=", 1)[1]))
            except ValueError:
                pass
            continue

        fields = [field.strip() for field in line.split(",")]
        if len(fields) not in (4, 7):
            continue
        name = fields[0]
        try:
            values = [float(field) for field in fields[1:]]
        except ValueError:
            continue
        translations[name] = (values[0], values[1], values[2])
        if len(values) == 6:
            rotations[name] = (values[3], values[4], values[5])
    return frame_number, translations, rotations


def map_object_names(values: dict) -> dict:
    """Rename streamed Vicon objects to the segment names used in training."""
    return {OBJECT_NAME_MAP.get(name, name): value for name, value in values.items()}


def open_socket(host: str, port: int) -> socket.socket:
    """Bind a non-blocking UDP socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.setblocking(False)
    return sock


def read_pending_frames(
    sock: socket.socket,
    fps: int,
    fallback_frame: int,
) -> list[LiveFrame]:
    """Drain every packet waiting on the socket and convert them to LiveFrames."""
    frames: list[LiveFrame] = []
    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            break

        frame_number, translations, rotations = parse_packet(data)
        if not translations:
            continue
        if frame_number is None:
            frame_number = fallback_frame + len(frames)
        frames.append(
            LiveFrame(
                frame_number=frame_number,
                timestamp_seconds=frame_number / fps,
                translations=map_object_names(translations),
                rotations=map_object_names(rotations),
            )
        )
    return frames


def space_was_pressed() -> tuple[bool, bool]:
    """Return (space_pressed, quit_requested) without blocking the loop."""
    if msvcrt is None:
        return False, False

    pressed = False
    quit_requested = False
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key == " ":
            pressed = True
        elif key in ("q", "Q", "\x1b"):
            quit_requested = True
    return pressed, quit_requested


def finish_segment(buffer, model, segment_index: int, min_frames: int) -> None:
    """Convert the buffered frames to a TrialRecord and classify them."""
    frame_count = len(buffer.frames)
    if frame_count < min_frames:
        print(f"Segment discarded: only {frame_count} frames (minimum {min_frames}).")
        return

    trial = buffer.to_trial_record(trial_name=f"live_segment_{segment_index:03d}")
    print(f"Segment recorded: {frame_count} frames ({frame_count / buffer.fps:.2f} s).")

    if model is None:
        return

    result = model.predict_trial(trial)
    print(
        "Prediction: "
        f"fPCA={result.fpca_prediction}, "
        f"statistical={result.statistical_prediction}, "
        f"agree={result.models_agree}"
    )


def run_probe(sock: socket.socket, fps: int) -> None:
    """Print incoming frames so the stream format can be verified."""
    print("Probe mode. Press Ctrl+C to quit.")
    last_reported = -1
    while True:
        for frame in read_pending_frames(sock, fps, last_reported + 1):
            last_reported = frame.frame_number
            print(f"frame {frame.frame_number}:")
            for name in sorted(frame.translations):
                tx, ty, tz = frame.translations[name]
                print(f"    {name}: ({tx:.1f}, {ty:.1f}, {tz:.1f})")


def main() -> None:
    """Run the UDP capture server."""
    args = parse_args()

    if msvcrt is None and not args.probe:
        raise RuntimeError("SPACE segmentation requires Windows (msvcrt).")

    sock = open_socket(args.host, args.port)
    print(f"Listening for Vicon UDP on {args.host}:{args.port} at {args.fps} fps.")

    try:
        if args.probe:
            run_probe(sock, args.fps)
            return

        model = load_or_train_model(args)
        print(f"Model: {args.model_path} (labels: {model.labels})")
        print(f"Required segments: {list(REQUIRED_SEGMENTS)}")
        print("Press SPACE to start/stop a segment. Press Q or Ctrl+C to quit.")

        buffer = LiveSegmentBuffer(fps=args.fps, required_segments=REQUIRED_SEGMENTS)
        recording = False
        segment_index = 1
        last_frame_number = -1
        skipped_frames = 0

        while True:
            pressed, quit_requested = space_was_pressed()
            if quit_requested:
                break
            if pressed:
                if not recording:
                    buffer.clear()
                    recording = True
                    skipped_frames = 0
                    print("Recording started.")
                else:
                    recording = False
                    if skipped_frames:
                        print(f"Skipped {skipped_frames} incomplete frames.")
                    finish_segment(buffer, model, segment_index, args.min_frames)
                    segment_index += 1

            for frame in read_pending_frames(sock, args.fps, last_frame_number + 1):
                if frame.frame_number == last_frame_number:
                    continue
                last_frame_number = frame.frame_number
                if not recording:
                    continue
                if any(name not in frame.translations for name in REQUIRED_SEGMENTS):
                    skipped_frames += 1
                    continue
                buffer.append(frame)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print("Stopped.")


if __name__ == "__main__":
    main()
