from config import RAW_DATA_DIR, create_project_directories


def main() -> None:
    """Glavna ulazna tačka aplikacije."""

    create_project_directories()

    print("Motion classification project")
    print(f"Folder sa originalnim CSV fajlovima: {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()