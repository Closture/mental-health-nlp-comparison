"""
9_cross_validation.py
5-fold stratified cross-validation for TF-IDF + LR and BiLSTM.
These are fast enough to cross-validate fully.

For the transformer models, we report the single train/val/test
split results with a note that cross-validation was computationally
prohibitive.
"""

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter

os.makedirs("results", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

N_FOLDS = 5
RANDOM_STATE = 42

print("Loading data...")
with open("data/splits.pkl", "rb") as f:
    splits = pickle.load(f)

# Combine train+val+test for proper CV
X_all = np.concatenate([splits["X_train"], splits["X_val"], splits["X_test"]])
y_all = np.concatenate([splits["y_train"], splits["y_val"], splits["y_test"]])
id2label = splits["id2label"]
LABELS = [id2label[i] for i in range(len(id2label))]
print(f"Total samples for CV: {len(X_all)}")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# ── 1. TF-IDF + LR Cross-Validation ──────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TF-IDF + LR  —  {N_FOLDS}-Fold Stratified CV")
print(f"{'='*50}")

tfidf_cv_scores = {"accuracy": [], "f1_macro": [], "precision": [], "recall": []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    X_tr, X_vl = X_all[train_idx], X_all[val_idx]
    y_tr, y_vl = y_all[train_idx], y_all[val_idx]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2),
            min_df=2, sublinear_tf=True, strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_tr, y_tr)
    preds = pipeline.predict(X_vl)

    acc  = accuracy_score(y_vl, preds)
    f1   = f1_score(y_vl, preds, average="macro")
    prec = precision_score(y_vl, preds, average="macro")
    rec  = recall_score(y_vl, preds, average="macro")

    tfidf_cv_scores["accuracy"].append(acc)
    tfidf_cv_scores["f1_macro"].append(f1)
    tfidf_cv_scores["precision"].append(prec)
    tfidf_cv_scores["recall"].append(rec)

    print(f"  Fold {fold}: Acc={acc:.4f}  F1={f1:.4f}  "
          f"Prec={prec:.4f}  Rec={rec:.4f}")

print(f"\n  Mean ± Std:")
for metric, vals in tfidf_cv_scores.items():
    print(f"    {metric:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

tfidf_cv_mean = {k: round(float(np.mean(v)), 4)
                 for k, v in tfidf_cv_scores.items()}
tfidf_cv_std  = {k: round(float(np.std(v)),  4)
                 for k, v in tfidf_cv_scores.items()}

# ── 2. BiLSTM Cross-Validation ────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"BiLSTM  —  {N_FOLDS}-Fold Stratified CV")
print(f"{'='*50}")

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM  = 128
HIDDEN_DIM = 256
N_LAYERS   = 2
DROPOUT    = 0.4
MAX_LEN    = 128
BATCH_SIZE = 64
EPOCHS     = 10
LR         = 1e-3
PATIENCE   = 3
NUM_LABELS = 4

print(f"Device: {DEVICE}")

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len):
        self.texts = texts; self.labels = labels
        self.word2idx = word2idx; self.max_len = max_len

    def encode(self, text):
        tokens = str(text).lower().split()[:self.max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return self.encode(self.texts[i]), torch.tensor(self.labels[i], dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        attn = torch.softmax(self.attention(out), dim=1)
        ctx  = (attn * out).sum(dim=1)
        return self.fc(self.dropout(ctx))

bilstm_cv_scores = {"accuracy": [], "f1_macro": [], "precision": [], "recall": []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all), 1):
    print(f"\n  Fold {fold}/{N_FOLDS}...")
    X_tr, X_vl = X_all[train_idx], X_all[val_idx]
    y_tr, y_vl = y_all[train_idx], y_all[val_idx]

    # Build vocab from this fold's training data only
    counter = Counter()
    for text in X_tr:
        counter.update(str(text).lower().split())
    vocab = ["<PAD>", "<UNK>"] + [w for w, c in counter.items() if c >= 2]
    word2idx = {w: i for i, w in enumerate(vocab)}
    VOCAB_SIZE = len(vocab)

    tr_loader = DataLoader(TextDataset(X_tr, y_tr, word2idx, MAX_LEN),
                           batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    vl_loader = DataLoader(TextDataset(X_vl, y_vl, word2idx, MAX_LEN),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM,
                             N_LAYERS, DROPOUT, NUM_LABELS).to(DEVICE)

    counts  = Counter(y_tr.tolist())
    weights = torch.tensor([1.0/counts[i] for i in range(NUM_LABELS)],
                           dtype=torch.float).to(DEVICE)
    weights = weights / weights.sum() * NUM_LABELS

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1, best_preds, patience_count = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            loss = criterion(model(Xb), yb)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        preds_list, labels_list = [], []
        with torch.no_grad():
            for Xb, yb in vl_loader:
                logits = model(Xb.to(DEVICE))
                preds_list.extend(logits.argmax(1).cpu().numpy())
                labels_list.extend(yb.numpy())

        val_f1 = f1_score(labels_list, preds_list, average="macro", zero_division=0)
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_preds = preds_list.copy()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    acc  = accuracy_score(y_vl, best_preds)
    f1   = f1_score(y_vl, best_preds, average="macro")
    prec = precision_score(y_vl, best_preds, average="macro")
    rec  = recall_score(y_vl, best_preds, average="macro")

    bilstm_cv_scores["accuracy"].append(acc)
    bilstm_cv_scores["f1_macro"].append(f1)
    bilstm_cv_scores["precision"].append(prec)
    bilstm_cv_scores["recall"].append(rec)

    print(f"    Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")

    del model; torch.cuda.empty_cache()

print(f"\n  Mean ± Std:")
for metric, vals in bilstm_cv_scores.items():
    print(f"    {metric:12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

bilstm_cv_mean = {k: round(float(np.mean(v)), 4)
                  for k, v in bilstm_cv_scores.items()}
bilstm_cv_std  = {k: round(float(np.std(v)),  4)
                  for k, v in bilstm_cv_scores.items()}

# ── 3. Save CV results ────────────────────────────────────────────────────────
cv_results = {
    "TF-IDF + LR": {
        "cv_mean": tfidf_cv_mean,
        "cv_std":  tfidf_cv_std,
        "fold_scores": {k: [round(v,4) for v in vals]
                        for k, vals in tfidf_cv_scores.items()}
    },
    "BiLSTM": {
        "cv_mean": bilstm_cv_mean,
        "cv_std":  bilstm_cv_std,
        "fold_scores": {k: [round(v,4) for v in vals]
                        for k, vals in bilstm_cv_scores.items()}
    },
}

with open("results/cv_results.json", "w") as f:
    json.dump(cv_results, f, indent=2)
print("\nSaved: results/cv_results.json")

# ── 4. CV comparison plot ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, (model_name, scores) in zip(axes, bilstm_cv_scores.items() if False else
    [("TF-IDF + LR", tfidf_cv_scores), ("BiLSTM", bilstm_cv_scores)]):

    folds = range(1, N_FOLDS + 1)
    ax.plot(folds, scores["accuracy"], "o-", label="Accuracy", color="#5B8DB8")
    ax.plot(folds, scores["f1_macro"], "s-", label="Macro F1", color="#D4756B")
    ax.axhline(np.mean(scores["f1_macro"]), linestyle="--",
               color="#D4756B", alpha=0.5,
               label=f"Mean F1={np.mean(scores['f1_macro']):.3f}")
    ax.set_title(f"{model_name} — 5-Fold CV", fontweight="bold", fontsize=12)
    ax.set_xlabel("Fold"); ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.0); ax.legend(fontsize=9)
    ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)

plt.suptitle("5-Fold Cross-Validation Results", fontweight="bold", fontsize=14)
plt.tight_layout()
plt.savefig("results/figures/cross_validation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/cross_validation.png")

# ── 5. Final summary table ────────────────────────────────────────────────────
print("\n── Cross-Validation Summary ──")
print(f"{'Model':<15} {'Acc Mean':>10} {'Acc Std':>8} "
      f"{'F1 Mean':>8} {'F1 Std':>8}")
print("-" * 55)
for name, mean, std in [
    ("TF-IDF + LR", tfidf_cv_mean, tfidf_cv_std),
    ("BiLSTM",      bilstm_cv_mean, bilstm_cv_std),
]:
    print(f"{name:<15} {mean['accuracy']:>10.4f} {std['accuracy']:>8.4f} "
          f"{mean['f1_macro']:>8.4f} {std['f1_macro']:>8.4f}")

print("\n✓ Cross-validation complete.")
print("\nNote: Transformer models (BERT, DistilBERT, RoBERTa) were not")
print("cross-validated due to computational cost (~107 mins per fold for BERT).")
