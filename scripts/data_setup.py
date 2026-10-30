import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_DATA_PATH = "data/enron_spam_data.csv"
load_dotenv(ENV_FILE)

REQUIRED_COLUMNS = {"Message ID", "Subject", "Message", "Spam/Ham"}
VALID_LABELS = {"spam", "ham"}


def normalize_for_duplicate_grouping(text: str) -> str:
    """Build a case- and whitespace-normalized key for duplicate grouping."""
    return " ".join(text.casefold().split())


def resolve_data_path(path: str | Path | None = None) -> Path:
    configured_path = path if path is not None else os.getenv(
        "DATASET_PATH", DEFAULT_DATA_PATH
    )
    if not str(configured_path).strip():
        raise ValueError("DATASET_PATH is empty")

    resolved_path = Path(configured_path).expanduser()
    if not resolved_path.is_absolute():
        resolved_path = PROJECT_ROOT / resolved_path
    return resolved_path.resolve()


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    path = resolve_data_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw_data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(raw_data.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {columns}")

    labels = raw_data["Spam/Ham"].astype("string").str.strip().str.lower()
    if labels.isna().any():
        raise ValueError("Dataset contains rows with a missing Spam/Ham label")

    unexpected_labels = set(labels.unique()).difference(VALID_LABELS)
    if unexpected_labels:
        values = ", ".join(sorted(unexpected_labels))
        raise ValueError(f"Dataset contains unexpected labels: {values}")

    subject = raw_data["Subject"].fillna("").astype(str).str.strip()
    message = raw_data["Message"].fillna("").astype(str).str.strip()

    return pd.DataFrame(
        {
            "message_id": raw_data["Message ID"],
            "text": (subject + "\n" + message).str.strip(),
            "label": labels,
        }
    )


def main() -> None:
    data = load_data()
    print(f"Loaded {len(data):,} messages from {resolve_data_path()}")
    print("Label counts:")
    print(data["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
