from config import RAW_DATASET_DIR, create_project_directories
from src.classification import (
    build_normalized_dataset,
    evaluate_leave_one_out_svm,
)
from src.data_loader import load_trials
from src.feature_extraction import extract_trial_features
from src.preprocessing import filter_trial_translations
from src.temporal_normalization import normalize_trial_features
from src.visualization import (
    plot_confusion_matrix,
    plot_normalized_trial_comparison,
    plot_original_vs_normalized_feature,
    plot_raw_vs_filtered_translation_side_by_side,
    plot_trial_features,
)


DEFAULT_SEGMENT_NAME = "Left:left"
DEFAULT_NORMALIZATION_SIGNAL = "distance_left_right"
COMPARISON_TRIAL_NAMES = ["Guranje_01", "Sirenje_01", "Sirenje_04"]


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

    normalized_features = normalize_trial_features(features, num_samples=101)
    print(
        "Prikazujem vremensku normalizaciju za signal "
        f"{DEFAULT_NORMALIZATION_SIGNAL}."
    )
    plot_original_vs_normalized_feature(
        features=features,
        normalized_features=normalized_features,
        signal_name=DEFAULT_NORMALIZATION_SIGNAL,
        show=True,
    )

    normalized_trials = [
        normalize_trial_features(
            extract_trial_features(filter_trial_translations(item, cutoff_hz=10.0)),
            num_samples=101,
        )
        for item in trials
    ]
    dataset = build_normalized_dataset(normalized_trials)
    result = evaluate_leave_one_out_svm(dataset)

    print(
        "SVM Leave-One-Out rezultat nad normalizovanim feature-ima: "
        f"accuracy={result.accuracy:.3f}"
    )
    for trial_name, true_label, predicted_label in zip(
        result.trial_names,
        result.y_true,
        result.y_pred,
    ):
        print(f"  {trial_name}: true={true_label}, predicted={predicted_label}")

    print("Prikazujem confusion matrix za SVM rezultat.")
    plot_confusion_matrix(result, show=True)

    normalized_by_name = {
        item.trial_name: item
        for item in normalized_trials
    }
    comparison_trials = [
        normalized_by_name[name]
        for name in COMPARISON_TRIAL_NAMES
        if name in normalized_by_name
    ]
    if len(comparison_trials) == len(COMPARISON_TRIAL_NAMES):
        print(
            "Prikazujem uporedni graf za "
            f"{', '.join(COMPARISON_TRIAL_NAMES)}."
        )
        plot_normalized_trial_comparison(comparison_trials, show=True)
    else:
        print("Preskacem uporedni graf jer nisu pronadjeni svi trazeni trial-i.")


if __name__ == "__main__":
    main()
