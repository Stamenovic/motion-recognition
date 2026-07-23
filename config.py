from pathlib import Path


# Glavni folder projekta
PROJECT_ROOT = Path(__file__).resolve().parent

# Folderi sa podacima
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DATASET_DIR = RAW_DATA_DIR / "DP2026"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ostali folderi
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_project_directories() -> None:
    """Kreira potrebne foldere ako oni ne postoje."""

    directories = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        RESULTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print("Projektni folderi su uspešno kreirani.")
