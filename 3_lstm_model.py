"""
3_lstm_model.py
Bidirectional LSTM with attention for the 4-class mental health task.
I added attention pooling because just using the last hidden state wasn't giving great results.
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from collections import Counter

os.makedirs("results/figures", exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 128
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.4
MAX_LEN = 128
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
PATIENCE = 3
NUM_LABELS = 4

print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load the splits we created earlier
with open("data/splits.pkl", "rb") as f:
    splits = pickle.load(f)

X_train = splits["X_train"]
y_train = splits["y_train"]
X_val   = splits["X_val"]
y_val   = splits["y_val"]
X_test  = splits["X_test"]
y_test  = splits["y_test"]

id2label = splits["id2label"]
LABELS   = [id2label[i] for i in range(NUM_LABELS)]

print("Building vocabulary...")
counter = Counter()
for text in X_train:
    counter.update(str(text).lower().split())

vocab = ["<PAD>", "<UNK>"] + [w for w, c in counter.items() if c >= 2]
word2idx = {w: i for i, w in enumerate(vocab)}
VOCAB_SIZE = len(vocab)
PAD_IDX = 0
UNK_IDX = 1
print(f"Vocab size: {VOCAB_SIZE:,}")

# Save vocabulary for later use
with open("results/lstm_vocab.pkl", "wb") as f:
    pickle.dump(word2idx, f)

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len

    def encode(self, text):
        tokens = str(text).lower().split()[:self.max_len]
        ids = [self.word2idx.get(t, UNK_IDX) for t in tokens]
        ids += [PAD_IDX] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.encode(self.texts[i]), torch.tensor(self.labels[i], dtype=torch.long)

# num_workers=0 is important on Windows - avoids multiprocessing errors
train_loader = DataLoader(TextDataset(X_train, y_train, word2idx, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_loader   = DataLoader(TextDataset(X_val, y_val, word2idx, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(TextDataset(X_test, y_test, word2idx, MAX_LEN),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        attn = torch.softmax(self.attention(lstm_out), dim=1)
        context = (attn * lstm_out).sum(dim=1)
        return self.fc(self.dropout(context))

model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT, NUM_LABELS).to(DEVICE)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class-balanced loss because the classes aren't perfectly balanced
counts = Counter(y_train.tolist())
weights = torch.tensor([1.0 / counts[i] for i in range(NUM_LABELS)], dtype=torch.float).to(DEVICE)
weights = weights / weights.sum() * NUM_LABELS

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * len(yb)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), acc, f1, np.array(all_preds), np.array(all_labels)

print("\nStarting BiLSTM training...")
history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
best_val_f1 = 0.0
best_epoch = 0
patience_count = 0

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc, tr_f1, _, _ = run_epoch(train_loader, train=True)
    vl_loss, vl_acc, vl_f1, _, _ = run_epoch(val_loader, train=False)
    scheduler.step(vl_f1)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_f1"].append(tr_f1)
    history["val_f1"].append(vl_f1)

    print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  "
          f"val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}")

    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        best_epoch = epoch
        patience_count = 0
        torch.save(model.state_dict(), "results/lstm_best.pt")
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

# Plot training curves
ep = range(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(ep, history["train_loss"], label="Train", color="#5B8DB8")
axes[0].plot(ep, history["val_loss"], label="Val", color="#D4756B")
axes[0].axvline(best_epoch, linestyle="--", color="gray", alpha=0.6)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(ep, history["train_f1"], label="Train", color="#5B8DB8")
axes[1].plot(ep, history["val_f1"], label="Val", color="#D4756B")
axes[1].axvline(best_epoch, linestyle="--", color="gray", alpha=0.6)
axes[1].set_title("Macro F1 Score")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.suptitle("BiLSTM Training Curves")
plt.tight_layout()
plt.savefig("results/figures/lstm_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved training curves.")

# Final test evaluation
print(f"\nLoading best model (epoch {best_epoch})...")
model.load_state_dict(torch.load("results/lstm_best.pt", map_location=DEVICE))
_, test_acc, test_f1, test_preds, test_labels = run_epoch(test_loader, train=False)
test_prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
test_rec  = recall_score(test_labels, test_preds, average="macro", zero_division=0)

print(f"\nTest — Accuracy: {test_acc:.4f}  Macro F1: {test_f1:.4f}")
print(classification_report(test_labels, test_preds, target_names=LABELS))

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=LABELS, yticklabels=LABELS)
ax.set_title("Confusion Matrix — BiLSTM")
ax.set_ylabel("True label")
ax.set_xlabel("Predicted label")
plt.tight_layout()
plt.savefig("results/figures/lstm_cm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/lstm_cm.png")

# Save results
results = {}
try:
    with open("results/all_results.json") as f:
        results = json.load(f)
except FileNotFoundError:
    pass

results["BiLSTM"] = {
    "accuracy": round(test_acc, 4),
    "f1_macro": round(test_f1, 4),
    "precision": round(test_prec, 4),
    "recall": round(test_rec, 4),
    "best_epoch": best_epoch,
}

with open("results/all_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nBiLSTM training finished and results saved.\n")