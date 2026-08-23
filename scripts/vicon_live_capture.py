"""Skeleton for Vicon streaming, SPACE-based segmentation, and prediction.

This file intentionally leaves the Vicon SDK calls as TODOs. The rest of the
flow mirrors the planned runtime architecture:

1. connect to Vicon,
2. continuously read frames,
3. SPACE starts/stops a segment buffer,
4. convert the finished buffer to TrialRecord,
5. classify it with the persisted live model.
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import MODELS_DIR
from src.feature_extraction import LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT
from src.live_capture import LiveFrame, LiveSegmentBuffer
from src.live_model import LiveMotionModel


MODEL_PATH = MODELS_DIR / "live_motion_model.joblib"
REQUIRED_SEGMENTS = (LEFT_SEGMENT, RIGHT_SEGMENT, TRUNK_SEGMENT)
FPS = 100


def main() -> None:
    """Run the live capture loop once Vicon SDK integration is filled in."""
    model = LiveMotionModel.load(MODEL_PATH)
    buffer = LiveSegmentBuffer(fps=FPS, required_segments=REQUIRED_SEGMENTS)
    recording = False
    segment_index = 1

    client = connect_to_vicon()
    print("Press SPACE to start/stop a segment. Press Ctrl+C to quit.")

    try:
        while True:
            frame = read_vicon_frame(client)
            if space_was_pressed():
                if not recording:
                    buffer.clear()
                    recording = True
                    print("Recording started.")
                else:
                    recording = False
                    trial = buffer.to_trial_record(
                        trial_name=f"live_segment_{segment_index:03d}",
                    )
                    segment_index += 1
                    result = model.predict_trial(trial)
                    print(
                        "Prediction: "
                        f"fPCA={result.fpca_prediction}, "
                        f"statistical={result.statistical_prediction}, "
                        f"agree={result.models_agree}"
                    )

            if recording:
                buffer.append(frame)
    except KeyboardInterrupt:
        print("Stopped.")


def connect_to_vicon():
    """Connect to Vicon DataStream SDK.

    TODO: Replace this placeholder with the concrete SDK client, host/IP and
    stream setup used in the lab.
    """
    raise NotImplementedError("Connect this function to Vicon DataStream SDK.")


def read_vicon_frame(client) -> LiveFrame:
    """Read and convert one Vicon frame.

    TODO: Use the SDK to read segment translations for:
    - Left:left
    - Right:right
    - Trup:trup

    Return them as a LiveFrame.
    """
    raise NotImplementedError("Convert one Vicon SDK frame to LiveFrame.")


def space_was_pressed() -> bool:
    """Return True when the user pressed SPACE.

    TODO: Hook this to a non-blocking keyboard listener. Keep this separate so
    the Vicon loop is not tied to one keyboard library.
    """
    return False


if __name__ == "__main__":
    main()
