"""Evaluate the saved ModernBERT and TF-IDF baselines on the held-out test split."""

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data_setup import PROJECT_ROOT, load_data
from train_ModernBERT import EmailDataset, OUTPUT_PATH as MODERNBERT_PATH
from train_TFIDF import build_model


SPLIT_PATH = PROJECT_ROOT / "outputs" / "split_manifest.csv"
LABELS = {"ham": 0, "spam": 1}


def report(name, actual, predicted):
    print(f"\n{name}")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["ham", "spam"],
            digits=4,
            zero_division=0,
        )
    )
    print("Confusion matrix (true rows, predicted columns; ham=0, spam=1):")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))


def main():
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError("Run scripts/make_splits.py first")
    if not MODERNBERT_PATH.is_dir():
        raise FileNotFoundError("Train ModernBERT before running this evaluation")

    data = load_data().merge(
        pd.read_csv(SPLIT_PATH),
        on="message_id",
        how="left",
        validate="one_to_one",
    )
    if data["split"].isna().any():
        raise ValueError("Some rows are missing from the split manifest")

    train_data = data[data["split"] == "train"]
    test_data = data[data["split"] == "test"]
    y_test = test_data["label"].map(LABELS).tolist()

    tfidf = build_model()
    tfidf.fit(train_data["text"], train_data["label"])
    tfidf_predictions = [
        LABELS[label] for label in tfidf.predict(test_data["text"])
    ]
    report("TF-IDF + Logistic Regression", y_test, tfidf_predictions)

    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_PATH)
    model = ModernBertForSequenceClassification.from_pretrained(MODERNBERT_PATH)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(PROJECT_ROOT / "outputs" / "test-evaluation"),
            per_device_eval_batch_size=6,
            report_to="none",
        ),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    predictions = trainer.predict(EmailDataset(test_data, tokenizer))
    y_pred = predictions.predictions.argmax(axis=1)
    report("ModernBERT", y_test, y_pred)


if __name__ == "__main__":
    main()
