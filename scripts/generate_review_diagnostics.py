"""Generate diagnostic plots requested during review."""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(PROJECT_ROOT / ".matplotlib-cache"),
)

from config import PLOTS_DIR, RAW_DATA_DIR, create_project_directories
from src.data_loader import load_trials
from src.feature_extraction import extract_trial_features
from src.fpca_features import build_fpca_dataset
from src.preprocessing import filter_trial_translations
from src.temporal_normalization import normalize_trial_features
from src.visualization import (
    plot_filter_cutoff_feature_comparison,
    plot_filter_cutoff_feature_grid,
    plot_filter_cutoff_feature_stacked,
    plot_filter_cutoff_translation_comparison,
    plot_fpca_score_scatter,
    plot_winter_residual_translation_analysis,
)


FILTER_ORDER = 2
FILTER_CUTOFF_HZ = 10.0
CUTOFFS_HZ = (6.0, 10.0, 15.0, 20.0)
NORMALIZED_NUM_SAMPLES = 101
SEGMENT_NAME = "Left:left"
WINTER_SEGMENT_NAMES = ("Left:left", "Right:right", "Trup:trup")
FEATURE_SIGNAL_NAME = "acceleration_left"
ADDITIONAL_FEATURE_SIGNAL_NAMES = ("speed_left",)
TRIAL_NAMES = ("Guranje_01", "Sirenje_01")
TRIAL_PAIR_INDICES = (1, 2, 3, 4)


def select_review_trials():
    """Select one representative trial for each reviewed movement type."""
    trials = load_trials(RAW_DATA_DIR)
    trials_by_name = {trial.trial_name: trial for trial in trials}
    missing = [name for name in TRIAL_NAMES if name not in trials_by_name]
    if missing:
        raise RuntimeError(f"Missing review trials: {missing}")
    return [trials_by_name[name] for name in TRIAL_NAMES], trials


def select_trial_pair(trials_by_name: dict, pair_index: int):
    """Select matching Guranje/Sirenje trials for one attempt index."""
    pair_names = (f"Guranje_{pair_index:02d}", f"Sirenje_{pair_index:02d}")
    missing = [name for name in pair_names if name not in trials_by_name]
    if missing:
        raise RuntimeError(f"Missing paired trials: {missing}")
    return [trials_by_name[name] for name in pair_names]


def build_normalized_trials(trials):
    """Run the same preprocessing and temporal normalization used by main.py."""
    filtered_trials = [
        filter_trial_translations(
            trial,
            cutoff_hz=FILTER_CUTOFF_HZ,
            order=FILTER_ORDER,
        )
        for trial in trials
    ]
    extracted_trials = [extract_trial_features(trial) for trial in filtered_trials]
    return [
        normalize_trial_features(features, num_samples=NORMALIZED_NUM_SAMPLES)
        for features in extracted_trials
    ]


def generate_pair_cutoff_plots(
    trials_by_name: dict,
    signal_name: str,
    output_dir: Path,
) -> list[Path]:
    """Generate one stacked cutoff plot for each Guranje/Sirenje trial pair."""
    paths = []
    for pair_index in TRIAL_PAIR_INDICES:
        pair_trials = select_trial_pair(trials_by_name, pair_index)
        paths.append(
            plot_filter_cutoff_feature_stacked(
                pair_trials,
                signal_name=signal_name,
                cutoff_hz_values=CUTOFFS_HZ,
                order=FILTER_ORDER,
                output_path=(
                    output_dir
                    / signal_name
                    / f"butterworth_cutoff_{signal_name}_pair_{pair_index:02d}.png"
                ),
                show=False,
            )
        )
    return paths


def safe_segment_file_name(segment_name: str) -> str:
    """Return a filesystem-friendly segment identifier."""
    return segment_name.replace(":", "_").replace(" ", "_").lower()


def generate_winter_residual_plots(
    trials,
    segment_names: tuple[str, ...],
    output_dir: Path,
) -> list[Path]:
    """Generate Winter-style residual plots for several segments."""
    paths = []
    for segment_name in segment_names:
        segment_file_name = safe_segment_file_name(segment_name)
        paths.append(
            plot_winter_residual_translation_analysis(
                trials,
                segment_name=segment_name,
                order=FILTER_ORDER,
                output_path=(
                    output_dir
                    / "winter_residual_segments"
                    / f"winter_residual_{segment_file_name}.png"
                ),
                show=False,
            )
        )
    return paths


def main() -> None:
    create_project_directories()
    selected_trials, all_trials = select_review_trials()
    trials_by_name = {trial.trial_name: trial for trial in all_trials}
    output_dir = PLOTS_DIR / "review_diagnostics"
    pair_output_dir = output_dir / "cutoff_pairs"

    cutoff_feature_path = plot_filter_cutoff_feature_comparison(
        selected_trials,
        signal_name=FEATURE_SIGNAL_NAME,
        cutoff_hz_values=CUTOFFS_HZ,
        order=FILTER_ORDER,
        output_path=output_dir / "01_butterworth_cutoff_acceleration_comparison.png",
        show=False,
    )
    cutoff_feature_grid_path = plot_filter_cutoff_feature_grid(
        selected_trials,
        signal_name=FEATURE_SIGNAL_NAME,
        cutoff_hz_values=CUTOFFS_HZ,
        order=FILTER_ORDER,
        output_path=output_dir / "02_butterworth_cutoff_acceleration_grid.png",
        show=False,
    )
    cutoff_feature_stacked_path = plot_filter_cutoff_feature_stacked(
        selected_trials,
        signal_name=FEATURE_SIGNAL_NAME,
        cutoff_hz_values=CUTOFFS_HZ,
        order=FILTER_ORDER,
        output_path=output_dir / "03_butterworth_cutoff_acceleration_stacked.png",
        show=False,
    )
    pair_cutoff_paths = generate_pair_cutoff_plots(
        trials_by_name,
        signal_name=FEATURE_SIGNAL_NAME,
        output_dir=pair_output_dir,
    )
    additional_feature_paths = []
    for signal_name in ADDITIONAL_FEATURE_SIGNAL_NAMES:
        additional_feature_paths.append(
            plot_filter_cutoff_feature_stacked(
                selected_trials,
                signal_name=signal_name,
                cutoff_hz_values=CUTOFFS_HZ,
                order=FILTER_ORDER,
                output_path=output_dir / f"07_butterworth_cutoff_{signal_name}.png",
                show=False,
            )
        )
        additional_feature_paths.extend(
            generate_pair_cutoff_plots(
                trials_by_name,
                signal_name=signal_name,
                output_dir=pair_output_dir,
            )
        )
    cutoff_translation_path = plot_filter_cutoff_translation_comparison(
        selected_trials,
        segment_name=SEGMENT_NAME,
        cutoff_hz_values=CUTOFFS_HZ,
        order=FILTER_ORDER,
        output_path=output_dir / "04_butterworth_cutoff_translation_comparison.png",
        show=False,
    )
    residual_path = plot_winter_residual_translation_analysis(
        selected_trials,
        segment_name=SEGMENT_NAME,
        order=FILTER_ORDER,
        output_path=output_dir / "05_winter_residual_translation_analysis.png",
        show=False,
    )
    winter_segment_paths = generate_winter_residual_plots(
        selected_trials,
        segment_names=WINTER_SEGMENT_NAMES,
        output_dir=output_dir,
    )

    fpca_dataset = build_fpca_dataset(
        build_normalized_trials(all_trials),
        n_components=2,
        standardize_signals=True,
    )
    fpca_path = plot_fpca_score_scatter(
        fpca_dataset,
        output_path=output_dir / "06_fpca_score_scatter.png",
        show=False,
    )

    for path in (
        cutoff_feature_path,
        cutoff_feature_grid_path,
        cutoff_feature_stacked_path,
        *pair_cutoff_paths,
        *additional_feature_paths,
        cutoff_translation_path,
        residual_path,
        *winter_segment_paths,
        fpca_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
