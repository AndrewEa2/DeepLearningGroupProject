"""Small helpers for the TF-IDF + Logistic Regression baseline."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_setup import PROJECT_ROOT, load_data

SPLIT_MANIFEST = PROJECT_ROOT / "outputs" / "split_manifest.csv"


def load_splits():
    data = load_data()
    splits = pd.read_csv(SPLIT_MANIFEST)
    data = data.merge(
        splits[["message_id", "split"]],
        on="message_id",
        how="left",
        validate="one_to_one",
    )
    if data["split"].isna().any():
        raise ValueError("Some dataset rows are missing from the split manifest")

    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    return train["text"], train["label"], validation["text"], validation["label"]


def build_model():
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=100_000)),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                    solver="liblinear",
                    verbose=1,
                ),
            ),
        ]
    )
