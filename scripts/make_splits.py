"""Create a reproducible train, validation, and test split manifest."""

import pandas as pd
from sklearn.model_selection import train_test_split

from data_setup import (
    PROJECT_ROOT,
    load_data,
    normalize_for_duplicate_grouping,
)


RANDOM_SEED = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "split_manifest.csv"


def main() -> None:
    data = load_data()

    if data["message_id"].isna().any() or data["message_id"].duplicated().any():
        raise ValueError("Message IDs must be present and unique")

    data["group"] = data["text"].map(normalize_for_duplicate_grouping)
    label_counts = data.groupby("group")["label"].nunique()
    if label_counts.gt(1).any():
        raise ValueError("Repeated text has conflicting labels")

    groups = data[["group", "label"]].drop_duplicates("group")
    train_validation, test = train_test_split(
        groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=groups["label"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_SEED,
        stratify=train_validation["label"],
    )

    group_to_split = {}
    for split_name, split_groups in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        for group in split_groups["group"]:
            group_to_split[group] = split_name

    manifest = pd.DataFrame(
        {
            "message_id": data["message_id"],
            "split": data["group"].map(group_to_split),
        }
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUTPUT_PATH, index=False)

    summary = (
        data.assign(split=manifest["split"])
        .groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
    )
    print(f"Saved split manifest to {OUTPUT_PATH}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
