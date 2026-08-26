"""Train and persist live-ready motion recognition models."""
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import MODELS_DIR, RAW_DATA_DIR, create_project_directories
from src.data_loader import load_trials
from src.live_model import train_live_statistical_model


MODEL_PATH = MODELS_DIR / "live_motion_model.joblib"
FILTER_CUTOFF_HZ = 10.0
FILTER_ORDER = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    """Train a statistical SVM model for completed live segments."""
    args = parse_args()
    create_project_directories()
    trials = load_trials(RAW_DATA_DIR)
    if not trials:
        raise RuntimeError(f"No labeled CSV trials found under {RAW_DATA_DIR}.")

    model = train_live_statistical_model(
        trials,
        cutoff_hz=FILTER_CUTOFF_HZ,
        filter_order=FILTER_ORDER,
    )
    model.save(args.model_path)

    print(f"Loaded trials: {len(trials)}")
    print("Model kind: statistical")
    print(f"Labels: {model.labels}")
    print(f"Unknown label: {model.unknown_label}")
    print(f"Unknown threshold: {model.unknown_threshold:.3f}")
    print(f"Minimum known motion: {model.minimum_motion_extent_mm:.1f} mm")
    print(f"Statistical features: {len(model.feature_names)}")
    print(f"Saved model: {args.model_path}")


if __name__ == "__main__":
    main()
