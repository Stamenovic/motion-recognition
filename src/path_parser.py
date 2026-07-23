"""Utilities for extracting trial labels and context from Vicon file paths."""
from pathlib import Path


LABEL_ALIASES = {
    "guranje": "guranje",
    "sirenje": "sirenje",
    "sirenje_ruku": "sirenje",
    "siri_ruke": "sirenje",
    "ispruzi_ruke": "guranje",
}


def parse_label_from_name(file_name: str) -> str | None:
    """Map a CSV file name such as Guranje_01.csv to a movement label."""
    stem = Path(file_name).stem.lower()

    for prefix, label in LABEL_ALIASES.items():
        if stem == prefix or stem.startswith(f"{prefix}_"):
            return label

    return None


def parse_trial_path(csv_path: Path, root: Path) -> dict[str, str] | None:
    """Extract patient, session, trial name and movement label from a CSV path."""
    label = parse_label_from_name(csv_path.name)
    if label is None:
        return None

    relative_parts = csv_path.relative_to(root).parts
    patient = relative_parts[0] if len(relative_parts) >= 3 else ""
    session = relative_parts[1] if len(relative_parts) >= 3 else ""

    return {
        "patient": patient,
        "session": session,
        "trial_name": csv_path.stem,
        "label": label,
    }
