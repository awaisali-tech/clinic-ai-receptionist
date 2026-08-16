import json
from pathlib import Path


def load_clinic_data(file_path: str | Path) -> dict:
    """
    Load clinic data from a JSON file.

    Args:
        file_path: Path to the clinic JSON file.

    Returns:
        The parsed JSON data as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON is invalid.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Clinic data file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return data
