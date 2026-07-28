"""Main entry point for the standardized motion-classification experiment."""
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

from config import PLOTS_DIR, RAW_DATA_DIR, create_project_directories
from src.classification import (
    ClassificationDataset,
    build_normalized_signal_dataset,
    build_statistical_dataset,
    evaluate_leave_one_out_svm,
)
from src.data_loader import load_trials
from src.feature_extraction import extract_trial_features
from src.fpca_features import build_fpca_dataset
from src.preprocessing import filter_trial_translations
from src.statistical_features import extract_statistical_features
from src.temporal_normalization import normalize_trial_features
from src.visualization import plot_confusion_matrix, plot_model_accuracy_comparison


FILTER_CUTOFF_HZ = 10.0
NORMALIZED_NUM_SAMPLES = 101
FPCA_COMPONENTS = 1
SAVE_PLOTS = False
SHOW_PLOTS = True


def fpca_to_classification_dataset(fpca_dataset) -> ClassificationDataset:
    """Adapt fPCA scores to the common SVM dataset structure."""
    return ClassificationDataset(
        X=fpca_dataset.X,
        y=fpca_dataset.y,
        trial_names=fpca_dataset.trial_names,
        feature_names=fpca_dataset.component_names,
    )


def print_result(name: str, result) -> None:
    """Print a compact classification report for one experiment."""
    print()
    print(name)
    print(f"Accuracy: {result.accuracy:.3f}")
    print("Confusion matrix:")
    print(result.confusion)
    for trial_name, true_label, predicted_label in zip(
        result.trial_names,
        result.y_true,
        result.y_pred,
    ):
        status = "OK" if true_label == predicted_label else "MISS"
        print(f"  {trial_name}: true={true_label}, predicted={predicted_label} [{status}]")


def run_experiment(
    show_plots: bool = SHOW_PLOTS,
    save_plots: bool = SAVE_PLOTS,
) -> dict[str, object]:
    """Run standardized SVM comparisons for the available Vicon CSV trials."""
    create_project_directories()

    trials = load_trials(RAW_DATA_DIR)
    if not trials:
        raise RuntimeError(f"No labeled CSV trials found under {RAW_DATA_DIR}.")

    print("Motion classification project")
    print(f"CSV root: {RAW_DATA_DIR}")
    print(f"Loaded trials: {len(trials)}")

    filtered_trials = [
        filter_trial_translations(trial, cutoff_hz=FILTER_CUTOFF_HZ)
        for trial in trials
    ]
    extracted_trials = [extract_trial_features(trial) for trial in filtered_trials]

    normalized_trials = [
        normalize_trial_features(features, num_samples=NORMALIZED_NUM_SAMPLES)
        for features in extracted_trials
    ]
    statistical_trials = [
        extract_statistical_features(features)
        for features in extracted_trials
    ]

    normalized_dataset = build_normalized_signal_dataset(normalized_trials)
    statistical_dataset = build_statistical_dataset(statistical_trials)
    fpca_dataset = build_fpca_dataset(
        normalized_trials,
        n_components=FPCA_COMPONENTS,
        standardize_signals=True,
    )

    datasets = {
        "standardized normalized signals": normalized_dataset,
        "statistical features": statistical_dataset,
        "standardized fPCA": fpca_to_classification_dataset(fpca_dataset),
    }
    results = {
        name: evaluate_leave_one_out_svm(dataset)
        for name, dataset in datasets.items()
    }

    print()
    print("Standardization:")
    print("- SVM: StandardScaler is fitted only on each training fold.")
    print("- fPCA: each normalized signal is z-score standardized before fPCA.")
    print(f"- fPCA explained variance ratio: {fpca_dataset.explained_variance_ratio}")

    for name, result in results.items():
        print_result(name, result)

    plot_dir = PLOTS_DIR / "main_standardized" if save_plots else None
    plot_model_accuracy_comparison(
        results,
        output_path=(plot_dir / "01_model_accuracy.png") if plot_dir else None,
        show=show_plots,
    )
    for index, (name, result) in enumerate(results.items(), start=2):
        safe_name = name.replace(" ", "_")
        plot_confusion_matrix(
            result,
            title=name,
            output_path=(plot_dir / f"{index:02d}_{safe_name}_confusion.png")
            if plot_dir
            else None,
            show=show_plots,
        )

    if save_plots:
        print(f"Saved plots to: {plot_dir}")

    return results


def main() -> None:
    """Glavna ulazna tacka aplikacije."""
    run_experiment()


if __name__ == "__main__":
    main()
