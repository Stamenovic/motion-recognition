from config import RAW_DATASET_DIR, create_project_directories
from src.classification import build_statistical_dataset, evaluate_leave_one_out_svm
from src.data_loader import load_trials
from src.feature_extraction import extract_trial_features
from src.preprocessing import filter_trial_translations
from src.statistical_features import extract_statistical_features
from src.visualization import (
    plot_accuracy_bar,
    plot_confusion_matrix,
    plot_statistical_feature_comparison,
)


COMPARISON_TRIAL_NAMES = ["Guranje_01", "Sirenje_01"]
COMPARISON_FEATURE_NAMES = [
    "distance_left_right_delta",
    "distance_left_right_range",
    "distance_left_right_positive_slope_ratio",
    "distance_left_right_time_to_max_ratio",
]


def print_classification_result(title: str, result) -> None:
    """Print a compact per-trial classification report."""
    print(f"{title}: accuracy={result.accuracy:.3f}")
    for trial_name, true_label, predicted_label in zip(
        result.trial_names,
        result.y_true,
        result.y_pred,
    ):
        print(f"  {trial_name}: true={true_label}, predicted={predicted_label}")


def main() -> None:
    """Run SVM classification without temporal normalization."""
    create_project_directories()

    print("Motion classification project")
    print(f"Folder sa DP2026 podacima: {RAW_DATASET_DIR}")

    trials = load_trials(RAW_DATASET_DIR)
    print(f"Broj ucitanih trial-a: {len(trials)}")
    if not trials:
        print("Nema ucitanih trial-a.")
        return

    statistical_trials = [
        extract_statistical_features(
            extract_trial_features(filter_trial_translations(trial, cutoff_hz=10.0))
        )
        for trial in trials
    ]
    dataset = build_statistical_dataset(statistical_trials)
    result = evaluate_leave_one_out_svm(dataset)

    print(f"Statistical dataset shape: {dataset.X.shape}")
    print_classification_result(
        "SVM bez vremenske normalizacije - statisticki feature-i",
        result,
    )

    print("Prikazujem accuracy graf.")
    plot_accuracy_bar(result, title="SVM without temporal normalization", show=True)
    print("Prikazujem confusion matrix.")
    plot_confusion_matrix(result, show=True)

    statistical_by_name = {item.trial_name: item for item in statistical_trials}
    comparison_trials = [
        statistical_by_name[name]
        for name in COMPARISON_TRIAL_NAMES
        if name in statistical_by_name
    ]
    if len(comparison_trials) == len(COMPARISON_TRIAL_NAMES):
        print(
            "Prikazujem statisticko poredjenje za "
            f"{', '.join(COMPARISON_TRIAL_NAMES)}."
        )
        plot_statistical_feature_comparison(
            comparison_trials,
            feature_names=COMPARISON_FEATURE_NAMES,
            show=True,
        )
    else:
        print("Preskacem poredjenje jer nisu pronadjeni svi trazeni trial-i.")


if __name__ == "__main__":
    main()
