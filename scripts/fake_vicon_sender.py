"""Fake Vicon Nexus UDP sender for testing the live capture server.

Streams synthetic Left/Right/Trup object poses in the same UDP format that
scripts/vicon_live_capture.py expects, so the whole live pipeline can be
tested without the lab.

Uses only the Python standard library, so it runs before the project
requirements are installed.

The stream alternates between a rest pose and a movement. Each phase change is
printed so you can compare sender timing with rolling-window predictions.
"""
import argparse
import math
import random
import socket
import struct
import time


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 51001
DEFAULT_FPS = 200  # the recorded CSVs in data/raw are 200 Hz

# Rest pose in millimetres (x = lateral, y = forward, z = up). The geometry
# below is tuned so the synthetic trials land in the same feature ranges as the
# recorded CSVs: hand separation 500 mm at rest, ~1490 mm at full sirenje,
# ~100 mm at full guranje, and right-hand lift for podizanje_desna.
REST_POSE = {
    "Trup": (0.0, 0.0, 1000.0),
    "Left": (-250.0, 150.0, 1250.0),
    "Right": (250.0, 150.0, 1250.0),
}
SPREAD_MM = 495.0  # sirenje: each hand moves outwards
CLOSE_MM = 200.0  # guranje: the hands converge as they extend
PUSH_MM = 400.0  # guranje: both hands travel forward
RIGHT_RAISE_MM = 450.0  # podizanje_desna: the right hand travels upward
NOISE_MM = 0.5

MOVEMENTS = ("sirenje", "guranje", "podizanje_desna")

_HEADER = struct.Struct("<IB")
_ITEM_HEADER = struct.Struct("<BH")
_ITEM_BODY = struct.Struct("<24s6d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_HOST, help="Target address.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Target UDP port.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Frames per second.")
    parser.add_argument(
        "--movement",
        choices=(*MOVEMENTS, "alternate"),
        default="alternate",
        help="Which movement to stream. 'alternate' switches every cycle.",
    )
    parser.add_argument(
        "--move-seconds",
        type=float,
        default=3.0,
        help="Duration of each movement. Recorded trials run 1.3-5.4 s.",
    )
    parser.add_argument(
        "--rest-seconds",
        type=float,
        default=2.0,
        help="Rest time between movements.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=0,
        help="Stop after this many movements. 0 means run until Ctrl+C.",
    )
    parser.add_argument(
        "--format",
        choices=("binary", "text"),
        default="binary",
        help="Packet encoding. 'binary' matches the Vicon UDP Object Stream.",
    )
    parser.add_argument(
        "--drop-rate",
        type=float,
        default=0.0,
        help="Probability (0-1) of omitting one object per frame, to simulate occlusion.",
    )
    parser.add_argument(
        "--noise-mm",
        type=float,
        default=NOISE_MM,
        help="Standard deviation of per-sample position noise.",
    )
    return parser.parse_args()


def movement_amplitude(progress: float) -> float:
    """Ramp up over the first half of the movement, then hold.

    The recorded trials do not return to the rest pose - sirenje ends with the
    hands apart and guranje ends with them extended - so the synthetic profile
    ramps to full amplitude and stays there.
    """
    if progress >= 0.5:
        return 1.0
    t = progress / 0.5
    return t * t * (3.0 - 2.0 * t)


def pose_at(movement: str, progress: float, noise_mm: float) -> dict:
    """Return object translations for one instant of a movement."""
    amplitude = movement_amplitude(progress) if progress is not None else 0.0
    spread = SPREAD_MM * amplitude if movement == "sirenje" else 0.0
    close = CLOSE_MM * amplitude if movement == "guranje" else 0.0
    push = PUSH_MM * amplitude if movement == "guranje" else 0.0
    right_raise = RIGHT_RAISE_MM * amplitude if movement == "podizanje_desna" else 0.0

    left_x, left_y, left_z = REST_POSE["Left"]
    right_x, right_y, right_z = REST_POSE["Right"]
    trunk_x, trunk_y, trunk_z = REST_POSE["Trup"]

    pose = {
        "Trup": (trunk_x, trunk_y, trunk_z),
        "Left": (left_x - spread + close, left_y + push, left_z),
        "Right": (right_x + spread - close, right_y + push, right_z + right_raise),
    }
    if noise_mm <= 0.0:
        return pose
    return {
        name: tuple(value + random.gauss(0.0, noise_mm) for value in translation)
        for name, translation in pose.items()
    }


def encode_binary(frame_number: int, pose: dict) -> bytes:
    """Encode one frame in the Vicon UDP Object Stream binary layout."""
    packet = _HEADER.pack(frame_number, len(pose))
    for name, (tx, ty, tz) in pose.items():
        body = _ITEM_BODY.pack(name.encode("ascii"), tx, ty, tz, 0.0, 0.0, 0.0)
        packet += _ITEM_HEADER.pack(0, len(body)) + body
    return packet


def encode_text(frame_number: int, pose: dict) -> bytes:
    """Encode one frame in the plain-text fallback format."""
    lines = [f"frame={frame_number}"]
    for name, (tx, ty, tz) in pose.items():
        lines.append(f"{name},{tx:.3f},{ty:.3f},{tz:.3f},0.000,0.000,0.000")
    return "\n".join(lines).encode("utf-8")


def apply_drop(pose: dict, drop_rate: float) -> dict:
    """Randomly omit one object to simulate a marker dropout."""
    if drop_rate <= 0.0 or random.random() >= drop_rate:
        return pose
    dropped = random.choice(list(pose))
    return {name: value for name, value in pose.items() if name != dropped}


def main() -> None:
    """Stream synthetic Vicon frames over UDP."""
    args = parse_args()
    encode = encode_binary if args.format == "binary" else encode_text
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    print(f"Sending {args.format} frames to {args.host}:{args.port} at {args.fps} fps.")
    print(f"Objects: {', '.join(REST_POSE)}")
    print("Press Ctrl+C to stop.")
    print()

    move_frames = max(int(args.move_seconds * args.fps), 1)
    rest_frames = max(int(args.rest_seconds * args.fps), 1)
    frame_number = 0
    cycle = 0
    next_send = time.perf_counter()

    try:
        while True:
            cycle += 1
            if args.cycles and cycle > args.cycles:
                break

            if args.movement == "alternate":
                movement = MOVEMENTS[(cycle - 1) % len(MOVEMENTS)]
            else:
                movement = args.movement

            print(f"[{cycle:03d}] rest {args.rest_seconds:.1f}s")
            for index in range(rest_frames):
                remaining = rest_frames - index
                if remaining % args.fps == 0:
                    print(f"        MOVE in {remaining // args.fps}s")
                pose = pose_at(movement, None, args.noise_mm)
                sock.sendto(encode(frame_number, apply_drop(pose, args.drop_rate)), target)
                frame_number += 1
                next_send += 1.0 / args.fps
                _sleep_until(next_send)

            print(f"[{cycle:03d}] MOVE {movement} ({args.move_seconds:.1f}s)")
            for index in range(move_frames):
                pose = pose_at(movement, index / move_frames, args.noise_mm)
                sock.sendto(encode(frame_number, apply_drop(pose, args.drop_rate)), target)
                frame_number += 1
                next_send += 1.0 / args.fps
                _sleep_until(next_send)

            print(f"[{cycle:03d}] done")
            print()
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        print(f"Stopped after {frame_number} frames.")


def _sleep_until(deadline: float) -> None:
    """Sleep until the deadline, skipping ahead if the loop already fell behind."""
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    main()
