"""Fine-tune ModernBERT for ham/spam classification."""

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    ModernBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from data_setup import PROJECT_ROOT, load_data


MODEL_NAME = "answerdotai/ModernBERT-base"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "split_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "modernbert-only"
LABELS = {"ham": 0, "spam": 1}


def load_splits():
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError("Run scripts/make_splits.py first")

    data = load_data().merge(
        pd.read_csv(SPLIT_PATH),
        on="message_id",
        how="left",
        validate="one_to_one",
    )
    if data["split"].isna().any():
        raise ValueError("Some rows are missing from the split manifest")
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    return train, validation


class EmailDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.tokens = tokenizer(data["text"].tolist(), truncation=True, max_length=512)
        self.labels = data["label"].map(LABELS).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.tokens.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def train():
    train_data, validation_data = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ModernBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "ham", 1: "spam"}, label2id=LABELS
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(OUTPUT_PATH),
            num_train_epochs=1,
            per_device_train_batch_size=6,
            learning_rate=2e-5,
            save_strategy="no",
            report_to="none",
            seed=42,
        ),
        train_dataset=EmailDataset(train_data, tokenizer),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    print(f"Using device: {trainer.args.device}")
    trainer.train()

    predictions = trainer.predict(EmailDataset(validation_data, tokenizer))
    y_true = validation_data["label"].map(LABELS).tolist()
    y_pred = predictions.predictions.argmax(axis=1)
    print(
        classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["ham", "spam"]
        )
    )
    print("Confusion matrix (ham=0, spam=1):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

    trainer.save_model(str(OUTPUT_PATH))
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
