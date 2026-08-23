"""Train and persist live-ready motion recognition models."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))

from config import MODELS_DIR, RAW_DATA_DIR, create_project_directories
from src.data_loader import load_trials
from src.live_model import train_live_motion_model


MODEL_PATH = MODELS_DIR / "live_motion_model.joblib"
FILTER_CUTOFF_HZ = 10.0
FILTER_ORDER = 2
NORMALIZED_NUM_SAMPLES = 101
FPCA_COMPONENTS = 2


def main() -> None:
    """Train fPCA/SVM and statistical/SVM models for completed live segments."""
    create_project_directories()
    trials = load_trials(RAW_DATA_DIR)
    if not trials:
        raise RuntimeError(f"No labeled CSV trials found under {RAW_DATA_DIR}.")

    model = train_live_motion_model(
        trials,
        cutoff_hz=FILTER_CUTOFF_HZ,
        filter_order=FILTER_ORDER,
        normalized_num_samples=NORMALIZED_NUM_SAMPLES,
        fpca_components=FPCA_COMPONENTS,
    )
    model.save(MODEL_PATH)

    print(f"Loaded trials: {len(trials)}")
    print(f"Labels: {model.labels}")
    print(f"fPCA components: {model.fpca_components}")
    print(f"Signals: {model.signal_names}")
    print(f"Statistical features: {len(model.statistical_feature_names)}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
