from config import RAW_DATASET_DIR, create_project_directories
from src.data_loader import load_trials
from src.feature_extraction import extract_trial_features
from src.preprocessing import filter_trial_translations
from src.visualization import (
    plot_raw_vs_filtered_translation_side_by_side,
    plot_trial_features,
)


DEFAULT_SEGMENT_NAME = "Left:left"


def main() -> None:
    """Glavna ulazna tačka aplikacije."""

    create_project_directories()

    print("Motion classification project")
    print(f"Folder sa DP2026 podacima: {RAW_DATASET_DIR}")

    trials = load_trials(RAW_DATASET_DIR)
    print(f"Broj ucitanih trial-a: {len(trials)}")

    if not trials:
        print("Nema ucitanih trial-a za plotovanje.")
        return

    trial = next((item for item in trials if item.trial_name == "Guranje_01"), trials[0])
    filtered_trial = filter_trial_translations(trial, cutoff_hz=10.0)
    print(
        "Prikazujem side-by-side preprocessing graf "
        f"za {trial.trial_name}, segment {DEFAULT_SEGMENT_NAME}."
    )
    plot_raw_vs_filtered_translation_side_by_side(
        raw_trial=trial,
        filtered_trial=filtered_trial,
        segment_name=DEFAULT_SEGMENT_NAME,
        show=True,
    )

    features = extract_trial_features(filtered_trial)
    print(f"Prikazujem izdvojene karakteristike za {features.trial_name}.")
    plot_trial_features(features, show=True)


if __name__ == "__main__":
    main()
