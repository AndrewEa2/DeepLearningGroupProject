import math
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ModernBertModel

from data_setup import PROJECT_ROOT, load_data


MODEL_NAME = "answerdotai/ModernBERT-base"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "split_manifest.csv"
OUTPUT_PATH = PROJECT_ROOT / "models" / "modernbert-gcn"
LABELS = {"ham": 0, "spam": 1}
MAX_LENGTH = 512
MIN_FREQUENCY = 2
MAX_VOCABULARY = 30000
WINDOW_SIZE = 5
GRAPH_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 2e-5
RANDOM_SEED = 42
TOKEN_PATTERN = re.compile(r"[a-z]+")


def load_splits():
    data = load_data().merge(
        pd.read_csv(SPLIT_PATH),
        on="message_id",
        how="left",
        validate="one_to_one",
    )
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]
    return train, validation


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def build_graph(train_texts):
    documents = [tokenize(text) for text in train_texts]
    frequencies = Counter(token for document in documents for token in document)
    vocabulary = {
        token: index
        for index, (token, frequency) in enumerate(
            (item for item in frequencies.most_common() if item[1] >= MIN_FREQUENCY)
        )
        if index < MAX_VOCABULARY
    }
    document_frequency = Counter(
        token for document in documents for token in set(document) if token in vocabulary
    )
    idf = np.ones(len(vocabulary), dtype=np.float32)
    for token, index in vocabulary.items():
        idf[index] = math.log((len(documents) + 1) / (document_frequency[token] + 1)) + 1

    pair_counts = Counter()
    for document in documents:
        ids = [vocabulary[token] for token in document if token in vocabulary]
        for left, source in enumerate(ids):
            for target in ids[left + 1 : left + WINDOW_SIZE + 1]:
                if source != target:
                    pair_counts[(min(source, target), max(source, target))] += 1

    row_counts = Counter()
    total = 2 * sum(pair_counts.values())
    for (source, target), count in pair_counts.items():
        row_counts[source] += count
        row_counts[target] += count

    edges = {(index, index): 1.0 for index in range(len(vocabulary))}
    for (source, target), count in pair_counts.items():
        pmi = math.log((count * total) / (row_counts[source] * row_counts[target]))
        if pmi > 0:
            edges[(source, target)] = pmi
            edges[(target, source)] = pmi

    degree = np.zeros(len(vocabulary), dtype=np.float32)
    for (source, _), weight in edges.items():
        degree[source] += weight
    indices = []
    values = []
    for (source, target), weight in edges.items():
        indices.append((source, target))
        values.append(weight / math.sqrt(degree[source] * degree[target]))
    adjacency = torch.sparse_coo_tensor(
        torch.tensor(indices, dtype=torch.long).T,
        torch.tensor(values, dtype=torch.float32),
        (len(vocabulary), len(vocabulary)),
    ).coalesce()
    return vocabulary, idf, adjacency


class EmailDataset(Dataset):
    def __init__(self, data, tokenizer, vocabulary, idf):
        self.encodings = tokenizer(
            data["text"].tolist(), truncation=True, max_length=MAX_LENGTH
        )
        self.labels = data["label"].map(LABELS).tolist()
        self.graph_tokens = []
        self.graph_weights = []
        for text in data["text"]:
            counts = Counter(token for token in tokenize(text) if token in vocabulary)
            ids = [vocabulary[token] for token in counts]
            weights = [counts[token] * idf[vocabulary[token]] for token in counts]
            self.graph_tokens.append(ids)
            self.graph_weights.append(weights)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.encodings["input_ids"][index],
            "attention_mask": self.encodings["attention_mask"][index],
            "graph_tokens": self.graph_tokens[index],
            "graph_weights": self.graph_weights[index],
            "labels": self.labels[index],
        }


def collate(batch, tokenizer):
    tokens = tokenizer.pad(
        [{"input_ids": row["input_ids"], "attention_mask": row["attention_mask"]} for row in batch],
        return_tensors="pt",
    )
    max_words = max(1, max(len(row["graph_tokens"]) for row in batch))
    graph_tokens = torch.zeros((len(batch), max_words), dtype=torch.long)
    graph_weights = torch.zeros((len(batch), max_words), dtype=torch.float32)
    for index, row in enumerate(batch):
        count = len(row["graph_tokens"])
        if count:
            graph_tokens[index, :count] = torch.tensor(row["graph_tokens"])
            graph_weights[index, :count] = torch.tensor(row["graph_weights"])
    labels = torch.tensor([row["labels"] for row in batch], dtype=torch.long)
    return tokens, graph_tokens, graph_weights, labels


class GraphConvolution(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, adjacency, features):
        return torch.sparse.mm(adjacency, self.linear(features))


class ModernBertGCN(nn.Module):
    def __init__(self, model_name, vocabulary_size):
        super().__init__()
        self.encoder = ModernBertModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.word_embeddings = nn.Embedding(vocabulary_size, GRAPH_SIZE)
        self.graph_layer_1 = GraphConvolution(GRAPH_SIZE, GRAPH_SIZE)
        self.graph_layer_2 = GraphConvolution(GRAPH_SIZE, GRAPH_SIZE)
        self.classifier = nn.Linear(hidden_size + GRAPH_SIZE, 2)

    def encode_graph(self, adjacency):
        graph_features = F.relu(
            self.graph_layer_1(adjacency, self.word_embeddings.weight)
        )
        return self.graph_layer_2(adjacency, graph_features)

    def forward(
        self, adjacency, input_ids, attention_mask, graph_tokens, graph_weights,
        graph_features=None,
    ):
        # Use mixed precision for the encoder; sparse GCN operations stay float32.
        with torch.autocast(
            device_type=input_ids.device.type, dtype=torch.float16, enabled=input_ids.is_cuda
        ):
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1)
        text_features = (encoded.last_hidden_state.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        if graph_features is None:
            graph_features = self.encode_graph(adjacency)
        pooled_graph = graph_features[graph_tokens]
        weights = graph_weights.unsqueeze(-1)
        pooled_graph = (pooled_graph * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        return self.classifier(torch.cat((text_features, pooled_graph), dim=1))


def run_epoch(model, loader, adjacency, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    # Validation weights are fixed, so compute the full graph only once.
    graph_features = None
    if not training:
        with torch.no_grad():
            graph_features = model.encode_graph(adjacency)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    actual = []
    predicted = []
    total_loss = 0.0
    for batch_number, (tokens, graph_tokens, graph_weights, labels) in enumerate(loader, 1):
        tokens = {key: value.to(device, non_blocking=True) for key, value in tokens.items()}
        graph_tokens = graph_tokens.to(device, non_blocking=True)
        graph_weights = graph_weights.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(
                adjacency,
                tokens["input_ids"],
                tokens["attention_mask"],
                graph_tokens,
                graph_weights,
                graph_features=graph_features,
            )
            loss = F.cross_entropy(logits, labels)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * len(labels)
        actual.extend(labels.cpu().tolist())
        predicted.extend(logits.argmax(dim=1).cpu().tolist())
        if batch_number == 1 or batch_number % 50 == 0 or batch_number == len(loader):
            elapsed = time.perf_counter() - started
            seconds_per_batch = elapsed / batch_number
            remaining = seconds_per_batch * (len(loader) - batch_number) / 60
            phase = "Training" if training else "Validation"
            memory = ""
            if device.type == "cuda":
                peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
                memory = f", peak allocated {peak_gb:.2f} GiB"
            print(
                f"{phase} batch {batch_number}/{len(loader)}: "
                f"{seconds_per_batch:.2f} s/batch, ETA {remaining:.1f} min{memory}",
                flush=True,
            )
    return total_loss / len(loader.dataset), actual, predicted


def train():
    torch.manual_seed(RANDOM_SEED)
    print("Loading train and validation data", flush=True)
    train_data, validation_data = load_splits()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Building word graph from training text", flush=True)
    vocabulary, idf, adjacency = build_graph(train_data["text"].tolist())
    print("Tokenizing messages", flush=True)
    training_set = EmailDataset(train_data, tokenizer, vocabulary, idf)
    validation_set = EmailDataset(validation_data, tokenizer, vocabulary, idf)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(
        training_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer),
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_set, batch_size=BATCH_SIZE,
        collate_fn=lambda batch: collate(batch, tokenizer),
        pin_memory=device.type == "cuda",
    )
    adjacency = adjacency.to(device)
    print("Loading ModernBERT and initializing GCN", flush=True)
    model = ModernBertGCN(MODEL_NAME, len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"Using device: {device}", flush=True)
    print(f"Batch size: {BATCH_SIZE}; encoder FP16 mixed precision: {device.type == 'cuda'}", flush=True)
    print(f"Graph vocabulary: {len(vocabulary):,} words; adjacency entries: {adjacency._nnz():,}", flush=True)
    for epoch in range(EPOCHS):
        train_loss, _, _ = run_epoch(model, train_loader, adjacency, device, optimizer, scaler)
        print(f"Epoch {epoch + 1}: train loss {train_loss:.4f}")

    _, actual, predicted = run_epoch(model, validation_loader, adjacency, device)
    print(classification_report(actual, predicted, labels=[0, 1], target_names=["ham", "spam"], digits=4))
    print("Confusion matrix (true rows, predicted columns; ham=0, spam=1):")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "idf": idf,
            "adjacency_indices": adjacency.indices().cpu(),
            "adjacency_values": adjacency.values().cpu(),
            "adjacency_size": adjacency.size(),
        },
        OUTPUT_PATH / "model.pt",
    )
    tokenizer.save_pretrained(OUTPUT_PATH)
    print(f"Saved model to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()
