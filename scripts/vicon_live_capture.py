"""UDP server that detects and classifies live Vicon Nexus movement segments.

Runtime flow:

1. listen for UDP packets streamed by Vicon Nexus,
2. parse each packet into per-object translations,
3. estimate the shared starting pose from recent rest frames,
4. start recording when hand displacement or speed passes the trigger threshold,
5. stop when movement is quiet long enough, or when the segment reaches max size,
6. classify the completed segment once, then clear the buffer and cooldown.

Only this file talks to the network and the keyboard. Everything downstream
reuses the same modules the offline training pipeline uses.
"""
import argparse
import math
import os
import socket
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import MODELS_DIR, RAW_DATA_DIR
from src.data_loader import load_trials
from src.feature_extraction import LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT
from src.live_capture import LiveFrame, LiveSegmentBuffer
from src.live_model import LiveMotionModel, train_live_motion_model

try:
    from pythonosc import osc_packet
except ImportError:  # pragma: no cover - handled at runtime with a clear warning
    osc_packet = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - this server is Windows-only
    msvcrt = None


MODEL_PATH = MODELS_DIR / "live_motion_model.joblib"
STREAM_REQUIRED_SEGMENTS = (LEFT_SEGMENT, RIGHT_SEGMENT)
MODEL_REQUIRED_SEGMENTS = (LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT)
SYNTHETIC_TRUNK_ROTATION = (0.0, 0.0, 0.0)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 7000
DEFAULT_FPS = 200  # must match the training CSVs in data/raw, which are 200 Hz
DEFAULT_MIN_FRAMES = 280
DEFAULT_COOLDOWN_FRAMES = 100
DEFAULT_BASELINE_FRAMES = 50
DEFAULT_START_DELTA_MM = 80.0
DEFAULT_START_SPEED_MM_S = 150.0
DEFAULT_STOP_SPEED_MM_S = 200.0
DEFAULT_STOP_QUIET_FRAMES = 30
DEFAULT_MAX_SEGMENT_FRAMES = 1000
DEFAULT_TRANSLATION_LOG_EVERY_SEC = 0.5

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
        help="Do not classify triggered segments shorter than this many frames.",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=DEFAULT_COOLDOWN_FRAMES,
        help="Ignore this many complete frames after a segment is classified.",
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=DEFAULT_BASELINE_FRAMES,
        help="Recent rest frames used to estimate the shared starting pose.",
    )
    parser.add_argument(
        "--start-delta-mm",
        type=float,
        default=DEFAULT_START_DELTA_MM,
        help="Start recording when hand displacement from baseline exceeds this.",
    )
    parser.add_argument(
        "--start-speed-mm-s",
        type=float,
        default=DEFAULT_START_SPEED_MM_S,
        help="Start recording when hand speed exceeds this.",
    )
    parser.add_argument(
        "--stop-speed-mm-s",
        type=float,
        default=DEFAULT_STOP_SPEED_MM_S,
        help="Count a recording frame as quiet when hand speed is below this.",
    )
    parser.add_argument(
        "--stop-quiet-frames",
        type=int,
        default=DEFAULT_STOP_QUIET_FRAMES,
        help="Stop recording after this many quiet frames.",
    )
    parser.add_argument(
        "--max-segment-frames",
        type=int,
        default=DEFAULT_MAX_SEGMENT_FRAMES,
        help="Force classification when the active segment reaches this length.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Print parsed packets and exit without loading the model. "
        "Use this first to confirm the stream format and object names.",
    )
    parser.add_argument(
        "--raw-probe",
        action="store_true",
        help="Print raw UDP packet sizes and parsed object names for stream debugging.",
    )
    parser.add_argument(
        "--status-every-sec",
        type=float,
        default=2.0,
        help="Print live stream diagnostics this often while waiting for a prediction. "
        "Use 0 to disable periodic status messages.",
    )
    parser.add_argument(
        "--translation-log-every-sec",
        type=float,
        default=DEFAULT_TRANSLATION_LOG_EVERY_SEC,
        help="Print current Left, Right, and Trup translations this often. "
        "Use 0 to disable translation logging.",
    )
    parser.add_argument(
        "--allow-missing-trup",
        action="store_true",
        help="Allow live testing with only Left and Right streams by filling Trup "
        "from the current hand midpoint.",
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
    parsed = _parse_osc_packet(data)
    if parsed is not None:
        return parsed
    parsed = _parse_binary_packet(data)
    if parsed is not None:
        return parsed
    return _parse_text_packet(data)


def _parse_osc_packet(data: bytes):
    """Parse one OSC message or bundle, or return None for non-OSC packets.

    Supported message shapes include:
        /vicon/<object>/translation <tx> <ty> <tz>
        /vicon/<object>/rotation <rx> <ry> <rz>
        /vicon/<object> <tx> <ty> <tz> [<rx> <ry> <rz>]
        /vicon/object <object> <tx> <ty> <tz> [<rx> <ry> <rz>]
        /vicon/frame <frame_number>

    The parser intentionally accepts several address spellings because Nexus OSC
    labels differ between stream presets and plugin versions.
    """
    if not data.startswith((b"/", b"#bundle\x00")):
        return None
    if osc_packet is None:
        return None

    translations: dict[str, tuple[float, float, float]] = {}
    rotations: dict[str, tuple[float, float, float]] = {}
    frame_number = None

    try:
        packet = osc_packet.OscPacket(data)
    except Exception:
        return None

    for message in _iter_osc_messages(packet):
        address = getattr(message, "address", "")
        params = list(getattr(message, "params", ()))
        path_parts = [part for part in address.split("/") if part]
        if not path_parts:
            continue

        leaf = _normalize_osc_token(path_parts[-1])
        if leaf in {"frame", "framenumber", "frame_number"}:
            parsed_frame = _first_numeric(params)
            if parsed_frame is not None:
                frame_number = int(parsed_frame)
            continue

        parsed = _parse_osc_pose_message(path_parts, params)
        if parsed is None:
            continue

        name, translation, rotation = parsed
        if translation is not None:
            translations[name] = translation
        if rotation is not None:
            rotations[name] = rotation

    return frame_number, translations, rotations


def _iter_osc_messages(packet) -> list[Any]:
    """Return python-osc messages from a decoded packet in bundle order."""
    messages = []
    for timed_message in getattr(packet, "messages", ()):
        message = getattr(timed_message, "message", timed_message)
        messages.append(message)
    return messages


def _parse_osc_pose_message(
    path_parts: list[str],
    params: list[Any],
) -> tuple[str, tuple[float, float, float] | None, tuple[float, float, float] | None] | None:
    """Normalize one OSC pose message into object translation and/or rotation."""
    leaf = _normalize_osc_token(path_parts[-1])
    numeric_values = _numeric_values(params)

    if params and isinstance(params[0], str):
        name = params[0].strip()
        numeric_values = _numeric_values(params[1:])
    elif leaf in _OSC_TRANSLATION_NAMES | _OSC_ROTATION_NAMES:
        if len(path_parts) < 2:
            return None
        name = path_parts[-2]
    else:
        name = path_parts[-1]

    if not name:
        return None

    if leaf in _OSC_TRANSLATION_NAMES:
        if len(numeric_values) < 3:
            return None
        return name, tuple(numeric_values[:3]), None
    if leaf in _OSC_ROTATION_NAMES:
        if len(numeric_values) < 3:
            return None
        return name, None, tuple(numeric_values[:3])
    if len(numeric_values) >= 6:
        return name, tuple(numeric_values[:3]), tuple(numeric_values[3:6])
    if len(numeric_values) >= 3:
        return name, tuple(numeric_values[:3]), None
    return None


_OSC_TRANSLATION_NAMES = {
    "globaltranslation",
    "position",
    "pos",
    "translation",
    "translationmm",
    "translation_mm",
    "xyz",
}
_OSC_ROTATION_NAMES = {
    "globalrotation",
    "globalrotationeulerxyz",
    "orientation",
    "rotation",
    "rotationeuler",
    "rotationeulerxyz",
    "rot",
}


def _normalize_osc_token(value: str) -> str:
    """Return a loose token key for matching common OSC path variants."""
    return value.replace("-", "").replace("_", "").lower()


def _numeric_values(values: list[Any]) -> list[float]:
    """Return float-convertible OSC arguments, preserving order."""
    numeric = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return numeric


def _first_numeric(values: list[Any]) -> float | None:
    """Return the first float-convertible OSC argument."""
    numeric = _numeric_values(values)
    return numeric[0] if numeric else None


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
    normalized_map = {
        _normalize_osc_token(stream_name): segment_name
        for stream_name, segment_name in OBJECT_NAME_MAP.items()
    }
    return {
        OBJECT_NAME_MAP.get(name, normalized_map.get(_normalize_osc_token(name), name)): value
        for name, value in values.items()
    }


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
    stats: dict | None = None,
    osc_state: dict[str, Any] | None = None,
    required_segments: tuple[str, ...] = MODEL_REQUIRED_SEGMENTS,
) -> list[LiveFrame]:
    """Drain every packet waiting on the socket and convert them to LiveFrames."""
    frames: list[LiveFrame] = []
    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            break

        if stats is not None:
            stats["udp_packets"] += 1
        frame_number, translations, rotations = parse_packet(data)
        translations = map_object_names(translations)
        rotations = map_object_names(rotations)

        if osc_state is not None and data.startswith((b"/", b"#bundle\x00")):
            completed_frame = _append_osc_packet_to_frame(
                osc_state,
                fps,
                fallback_frame + len(frames),
                frame_number,
                translations,
                rotations,
                required_segments,
            )
            if completed_frame is not None:
                if stats is not None:
                    stats["parsed_frames"] += 1
                frames.append(completed_frame)
            elif not translations:
                if stats is not None:
                    stats["empty_packets"] += 1
                    osc_addresses = _parse_osc_addresses(data)
                    if osc_addresses:
                        stats["osc_packets"] += 1
                        stats["last_osc"] = ", ".join(osc_addresses[:5])
            continue

        if not translations:
            if stats is not None:
                stats["empty_packets"] += 1
                osc_addresses = _parse_osc_addresses(data)
                if osc_addresses:
                    stats["osc_packets"] += 1
                    stats["last_osc"] = ", ".join(osc_addresses[:5])
            continue
        if frame_number is None:
            frame_number = fallback_frame + len(frames)
        if stats is not None:
            stats["parsed_frames"] += 1
        frames.append(
            LiveFrame(
                frame_number=frame_number,
                timestamp_seconds=frame_number / fps,
                translations=translations,
                rotations=rotations,
            )
        )
    return frames


def _append_osc_packet_to_frame(
    state: dict[str, Any],
    fps: int,
    fallback_frame: int,
    frame_number: int | None,
    translations: dict[str, tuple[float, float, float]],
    rotations: dict[str, tuple[float, float, float]],
    required_segments: tuple[str, ...],
) -> LiveFrame | None:
    """Accumulate separate OSC messages until one complete live frame is ready."""
    state.setdefault("translations", {})
    state.setdefault("rotations", {})
    state.setdefault("frame_number", None)

    if frame_number is not None:
        current_frame_number = state.get("frame_number")
        if current_frame_number is not None and frame_number != current_frame_number:
            state["translations"] = {}
            state["rotations"] = {}
        state["frame_number"] = frame_number

    state["translations"].update(translations)
    state["rotations"].update(rotations)

    if any(name not in state["translations"] for name in required_segments):
        return None

    completed_frame_number = state["frame_number"]
    if completed_frame_number is None:
        completed_frame_number = fallback_frame

    completed_frame = LiveFrame(
        frame_number=completed_frame_number,
        timestamp_seconds=completed_frame_number / fps,
        translations=dict(state["translations"]),
        rotations=dict(state["rotations"]),
    )
    state["translations"] = {}
    state["rotations"] = {}
    state["frame_number"] = None
    return completed_frame


def fill_missing_trup_from_hands(frame: LiveFrame) -> LiveFrame:
    """Fill missing Trup with a stable hand midpoint for temporary live testing."""
    if TRUNK_SEGMENT in frame.translations:
        return frame
    if LEFT_SEGMENT not in frame.translations or RIGHT_SEGMENT not in frame.translations:
        return frame

    left = frame.translations[LEFT_SEGMENT]
    right = frame.translations[RIGHT_SEGMENT]
    midpoint = tuple((left_value + right_value) / 2.0 for left_value, right_value in zip(left, right))
    frame.translations[TRUNK_SEGMENT] = midpoint
    frame.rotations.setdefault(TRUNK_SEGMENT, SYNTHETIC_TRUNK_ROTATION)
    return frame


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


def classify_segment(buffer, model, event_index: int, min_frames: int):
    """Convert the triggered buffer to a TrialRecord and classify it."""
    frame_count = len(buffer.frames)
    if frame_count < min_frames:
        return None

    trial = buffer.to_trial_record(trial_name=f"live_segment_{event_index:06d}")
    result = model.predict_trial(trial)
    return trial, result


def _translation(frame: LiveFrame, segment_name: str) -> tuple[float, float, float]:
    """Return one segment translation from a live frame."""
    return frame.translations[segment_name]


def _mean_translation(
    frames: list[LiveFrame],
    segment_name: str,
) -> tuple[float, float, float]:
    """Return the mean translation for one segment over baseline frames."""
    values = [_translation(frame, segment_name) for frame in frames]
    return tuple(sum(axis_values) / len(values) for axis_values in zip(*values))


def _fit_baseline(frames: list[LiveFrame]) -> dict[str, tuple[float, float, float]]:
    """Estimate the shared starting pose from recent rest frames."""
    return {
        segment_name: _mean_translation(frames, segment_name)
        for segment_name in MODEL_REQUIRED_SEGMENTS
    }


def _max_hand_delta_from_baseline(
    frame: LiveFrame,
    baseline: dict[str, tuple[float, float, float]],
) -> float:
    """Return the larger left/right hand displacement from baseline."""
    return max(
        math.dist(_translation(frame, LEFT_SEGMENT), baseline[LEFT_SEGMENT]),
        math.dist(_translation(frame, RIGHT_SEGMENT), baseline[RIGHT_SEGMENT]),
    )


def _max_hand_speed(
    previous_frame: LiveFrame | None,
    frame: LiveFrame,
    fps: int,
) -> float:
    """Return the larger left/right hand speed in mm/s."""
    if previous_frame is None:
        return 0.0
    return max(
        math.dist(_translation(frame, LEFT_SEGMENT), _translation(previous_frame, LEFT_SEGMENT))
        * fps,
        math.dist(_translation(frame, RIGHT_SEGMENT), _translation(previous_frame, RIGHT_SEGMENT))
        * fps,
    )


def print_segment_prediction(event_index: int, trial, result, reason: str) -> None:
    """Print one classified triggered segment."""
    print(
        f"Detected movement {event_index}: "
        f"label={result.fpca_prediction}, "
        f"frames={trial.metadata.num_frames}, "
        f"range={trial.metadata.frame_start}-{trial.metadata.frame_end}, "
        f"candidate={result.known_prediction}, "
        f"confidence={result.fpca_confidence:.3f}, "
        f"motion={result.motion_extent_mm:.1f} mm, "
        f"reason={reason}"
    )


def _format_translation(frame: LiveFrame, segment_name: str) -> str:
    """Return a compact translation string for one required segment."""
    tx, ty, tz = _translation(frame, segment_name)
    stream_name = next(
        (name for name, mapped_name in OBJECT_NAME_MAP.items() if mapped_name == segment_name),
        segment_name,
    )
    return f"{stream_name}=({tx:.1f}, {ty:.1f}, {tz:.1f})"


def print_current_translations(frame: LiveFrame) -> None:
    """Print current mapped translations for the required live segments."""
    values = ", ".join(_format_translation(frame, segment) for segment in MODEL_REQUIRED_SEGMENTS)
    print(f"Translations frame={frame.frame_number}: {values}")


def run_probe(sock: socket.socket, fps: int) -> None:
    """Print incoming frames so the stream format can be verified."""
    print("Probe mode. Press Ctrl+C to quit.")
    last_reported = -1
    osc_state: dict[str, Any] = {}
    while True:
        for frame in read_pending_frames(
            sock,
            fps,
            last_reported + 1,
            osc_state=osc_state,
            required_segments=STREAM_REQUIRED_SEGMENTS,
        ):
            fill_missing_trup_from_hands(frame)
            last_reported = frame.frame_number
            print(f"frame {frame.frame_number}:")
            for name in sorted(frame.translations):
                tx, ty, tz = frame.translations[name]
                print(f"    {name}: ({tx:.1f}, {ty:.1f}, {tz:.1f})")


def run_raw_probe(sock: socket.socket) -> None:
    """Print raw UDP packet diagnostics before requiring a known packet format."""
    print("Raw probe mode. Press Ctrl+C to quit.")
    last_wait_message = 0.0
    packet_count = 0
    while True:
        try:
            data, address = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            now = time.monotonic()
            if now - last_wait_message >= 2.0:
                print("waiting for UDP packets...")
                last_wait_message = now
            time.sleep(0.02)
            continue

        packet_count += 1
        parsed = parse_packet(data)
        if parsed is None:
            frame_number, translations, rotations = None, {}, {}
        else:
            frame_number, translations, rotations = parsed
        preview = data[:24].hex(" ")
        names = ", ".join(sorted(translations)) if translations else "no parsed objects"
        osc_addresses = _parse_osc_addresses(data)
        osc_text = f" osc={', '.join(osc_addresses)}" if osc_addresses else ""
        print(
            f"packet {packet_count} from {address[0]}:{address[1]} "
            f"bytes={len(data)} frame={frame_number} objects={names} "
            f"first24={preview}{osc_text}"
        )


def _parse_osc_addresses(data: bytes) -> list[str]:
    """Return OSC message addresses from one OSC packet or bundle."""
    if not data.startswith((b"/", b"#bundle\x00")) or osc_packet is None:
        return []
    try:
        packet = osc_packet.OscPacket(data)
    except Exception:
        return []
    return [message.address for message in _iter_osc_messages(packet)]


def main() -> None:
    """Run the UDP capture server."""
    args = parse_args()

    if args.min_frames < 2:
        raise ValueError("--min-frames must be at least 2.")
    if args.cooldown_frames < 0:
        raise ValueError("--cooldown-frames cannot be negative.")
    if args.baseline_frames < 1:
        raise ValueError("--baseline-frames must be at least 1.")
    if args.stop_quiet_frames < 1:
        raise ValueError("--stop-quiet-frames must be at least 1.")
    if args.max_segment_frames < args.min_frames:
        raise ValueError("--max-segment-frames must be greater than or equal to --min-frames.")
    if args.status_every_sec < 0:
        raise ValueError("--status-every-sec cannot be negative.")
    if args.translation_log_every_sec < 0:
        raise ValueError("--translation-log-every-sec cannot be negative.")

    sock = open_socket(args.host, args.port)
    print(f"Listening for Vicon UDP on {args.host}:{args.port} at {args.fps} fps.")
    if osc_packet is None:
        print("python-osc is not installed; OSC packets will be ignored.")
    skipped_frames = 0

    try:
        if args.raw_probe:
            run_raw_probe(sock)
            return

        if args.probe:
            run_probe(sock, args.fps)
            return

        model = load_or_train_model(args)
        stream_required_segments = (
            STREAM_REQUIRED_SEGMENTS if args.allow_missing_trup else MODEL_REQUIRED_SEGMENTS
        )
        print(f"Model: {args.model_path} (labels: {model.labels})")
        print(f"Unknown label: {model.unknown_label} (threshold={model.unknown_threshold:.3f})")
        print(f"Minimum motion for known label: {model.minimum_motion_extent_mm:.1f} mm")
        print(f"Required stream segments: {list(stream_required_segments)}")
        print(f"Required model segments: {list(MODEL_REQUIRED_SEGMENTS)}")
        if args.allow_missing_trup:
            print("Missing Trup will be filled from the Left/Right midpoint for this run.")
        print(f"Baseline frames: {args.baseline_frames}")
        print(f"Start delta: {args.start_delta_mm:.1f} mm")
        print(f"Start speed: {args.start_speed_mm_s:.1f} mm/s")
        print(f"Stop speed: {args.stop_speed_mm_s:.1f} mm/s")
        print(f"Stop quiet frames: {args.stop_quiet_frames}")
        print(f"Minimum segment frames: {args.min_frames}")
        print(f"Maximum segment frames: {args.max_segment_frames}")
        print(f"Cooldown after segment: {args.cooldown_frames} complete frame(s)")
        print(f"Status interval: {args.status_every_sec:.1f} s")
        print(f"Translation log interval: {args.translation_log_every_sec:.1f} s")
        print("Press Q or Ctrl+C to quit.")

        buffer = LiveSegmentBuffer(fps=args.fps, required_segments=MODEL_REQUIRED_SEGMENTS)
        event_index = 1
        last_frame_number = -1
        cooldown_frames_remaining = 0
        baseline_frames: list[LiveFrame] = []
        previous_frame = None
        quiet_count = 0
        recording = False
        latest_complete_frame = None
        stream_stats = {
            "udp_packets": 0,
            "parsed_frames": 0,
            "empty_packets": 0,
            "osc_packets": 0,
            "last_osc": "",
        }
        osc_state: dict[str, Any] = {}
        last_status_time = time.monotonic()
        last_status_snapshot = (0, 0, 0, 0, 0, False)
        last_translation_log_time = time.monotonic()

        while True:
            pressed, quit_requested = space_was_pressed()
            if quit_requested:
                break
            if pressed:
                print("SPACE is ignored in trigger mode. Press Q to quit.")

            pending_frames = read_pending_frames(
                sock,
                args.fps,
                last_frame_number + 1,
                stream_stats,
                osc_state,
                stream_required_segments,
            )
            for frame in pending_frames:
                if frame.frame_number == last_frame_number:
                    continue
                last_frame_number = frame.frame_number
                if args.allow_missing_trup:
                    fill_missing_trup_from_hands(frame)
                if any(name not in frame.translations for name in MODEL_REQUIRED_SEGMENTS):
                    skipped_frames += 1
                    continue
                latest_complete_frame = frame
                speed = _max_hand_speed(previous_frame, frame, args.fps)
                previous_frame = frame
                if cooldown_frames_remaining:
                    cooldown_frames_remaining -= 1
                    continue

                if not recording:
                    baseline_frames.append(frame)
                    if len(baseline_frames) > args.baseline_frames:
                        baseline_frames = baseline_frames[-args.baseline_frames:]
                    if len(baseline_frames) < args.baseline_frames:
                        continue

                    baseline = _fit_baseline(baseline_frames)
                    hand_delta = _max_hand_delta_from_baseline(frame, baseline)
                    if (
                        hand_delta < args.start_delta_mm
                        and speed < args.start_speed_mm_s
                    ):
                        continue

                    recording = True
                    buffer.clear()
                    buffer.append(frame)
                    quiet_count = 0
                    continue

                buffer.append(frame)
                if speed <= args.stop_speed_mm_s:
                    quiet_count += 1
                else:
                    quiet_count = 0

                quiet_stop = (
                    len(buffer.frames) >= args.min_frames
                    and quiet_count >= args.stop_quiet_frames
                )
                forced_stop = len(buffer.frames) >= args.max_segment_frames
                if quiet_stop:
                    stop_reason = "quiet"
                elif forced_stop:
                    stop_reason = "max"
                else:
                    continue

                classified = classify_segment(
                    buffer,
                    model,
                    event_index,
                    args.min_frames,
                )
                if classified is None:
                    recording = False
                    buffer.clear()
                    baseline_frames = []
                    quiet_count = 0
                    continue
                trial, result = classified
                print_segment_prediction(event_index, trial, result, stop_reason)
                event_index += 1
                buffer.clear()
                baseline_frames = []
                quiet_count = 0
                recording = False
                cooldown_frames_remaining = args.cooldown_frames

            if args.status_every_sec:
                now = time.monotonic()
                if now - last_status_time >= args.status_every_sec:
                    snapshot = (
                        stream_stats["udp_packets"],
                        stream_stats["parsed_frames"],
                        stream_stats["empty_packets"],
                        stream_stats["osc_packets"],
                        skipped_frames,
                        recording,
                    )
                    if snapshot != last_status_snapshot:
                        if stream_stats["udp_packets"] == 0:
                            print("Waiting for UDP packets...")
                        elif stream_stats["parsed_frames"] == 0:
                            if stream_stats["osc_packets"]:
                                print(
                                    "Receiving OSC packets, but no object translations were parsed "
                                    f"(last OSC: {stream_stats['last_osc']}). "
                                    "Use --raw-probe to inspect the stream, or switch Nexus to the "
                                    "Object Stream layout expected by this script."
                                )
                            else:
                                print(
                                    "Receiving UDP packets, but no object translations were parsed. "
                                    "Use --raw-probe to inspect the packet format."
                                )
                        elif skipped_frames:
                            print(
                                f"Parsed {stream_stats['parsed_frames']} frame(s), "
                                f"but skipped {skipped_frames} incomplete frame(s). "
                                f"Need segments: {list(stream_required_segments)}."
                            )
                        elif recording:
                            print(
                                f"Recording movement: {len(buffer.frames)} frame(s), "
                                f"quiet={quiet_count}/{args.stop_quiet_frames}."
                            )
                        else:
                            print(
                                f"Ready: parsed {stream_stats['parsed_frames']} complete frame(s). "
                                "Move past the start threshold to trigger classification."
                            )
                    last_status_snapshot = snapshot
                    last_status_time = now
            if args.translation_log_every_sec and latest_complete_frame is not None:
                now = time.monotonic()
                if now - last_translation_log_time >= args.translation_log_every_sec:
                    print_current_translations(latest_complete_frame)
                    last_translation_log_time = now
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        if skipped_frames:
            print(f"Skipped {skipped_frames} incomplete frames.")
        print("Stopped.")


if __name__ == "__main__":
    main()
