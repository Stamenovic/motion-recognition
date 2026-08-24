"""Utilities for extracting trial labels and context from Vicon file paths."""
from pathlib import Path
import re


LABEL_ALIASES = {
    "guranje": "guranje",
    "sirenje": "sirenje",
    "siri_ruke": "sirenje",
    "ispruzi_ruke": "guranje",
    "podizanje_desna": "podizanje_desna",
}


def _normalize_name_part(value: str) -> str:
    """Remove common separators so names match with or without underscores."""
    return re.sub(r"[_\-\s]+", "", value.lower())


def parse_label_from_name(file_name: str) -> str | None:
    """Map CSV names such as Guranje01.csv or Podizanje_desna_01.csv to labels."""
    stem = Path(file_name).stem.lower()
    normalized_stem = _normalize_name_part(stem)
    for prefix, label in LABEL_ALIASES.items():
        normalized_prefix = _normalize_name_part(prefix)
        suffix = normalized_stem.removeprefix(normalized_prefix)
        if normalized_stem.startswith(normalized_prefix) and (
            suffix == "" or suffix[0].isdigit()
        ):
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
