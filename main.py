from config import RAW_DATASET_DIR, create_project_directories


def main() -> None:
    """Glavna ulazna tačka aplikacije."""

    create_project_directories()

    print("Motion classification project")
    print(f"Folder sa DP2026 podacima: {RAW_DATASET_DIR}")


if __name__ == "__main__":
    main()
