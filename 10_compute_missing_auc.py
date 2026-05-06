"""
10_compute_missing_auc.py
Computes ROC AUC (macro OvR) for BiLSTM and Random DistilBERT
by reloading each saved model and running inference on the test set
to get probability outputs.
"""

import os, json, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = 4
print(f"Device: {DEVICE}")

# Load test set and label info for AUC calculations
with open("data/splits.pkl", "rb") as f:
    splits = pickle.load(f)
X_test   = splits["X_test"]
y_test   = splits["y_test"]
id2label = splits["id2label"]
LABELS   = [id2label[i] for i in range(NUM_LABELS)]

y_bin = label_binarize(y_test, classes=list(range(NUM_LABELS)))

# Load existing results to update with AUC scores
with open("results/all_results.json") as f:
    all_results = json.load(f)

# We will compute AUC for BiLSTM and Random DistilBERT, since TF-IDF AUC was computed in the baseline script and the other transformers were computed in their respective scripts.
print("\nComputing BiLSTM AUC...")

# BiLSTM model definition (same as in 3_lstm_model.py)
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=128):
        self.texts    = texts
        self.labels   = labels
        self.word2idx = word2idx
        self.max_len  = max_len

    def encode(self, text):
        tokens = str(text).lower().split()[:self.max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        ids   += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):  return len(self.labels)
    def __getitem__(self, i):
        return self.encode(self.texts[i]), torch.tensor(self.labels[i], dtype=torch.long)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 n_layers=2, dropout=0.4, n_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        emb         = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(emb)
        attn        = torch.softmax(self.attention(lstm_out), dim=1)
        context     = (attn * lstm_out).sum(dim=1)
        return self.fc(self.dropout(context))

try:
    with open("results/lstm_vocab.pkl", "rb") as f:
        word2idx = pickle.load(f)

    model = BiLSTMClassifier(vocab_size=len(word2idx)).to(DEVICE)
    model.load_state_dict(torch.load("results/lstm_best.pt",
                                      map_location=DEVICE,
                                      weights_only=False))
    model.eval()

    loader = DataLoader(TextDataset(X_test, y_test, word2idx),
                        batch_size=64, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for Xb, _ in loader:
            logits = model(Xb.to(DEVICE))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)

    probs_arr = np.array(all_probs)
    auc = roc_auc_score(y_bin, probs_arr, average="macro", multi_class="ovr")
    all_results["BiLSTM"]["auc"] = round(auc, 4)
    print(f"  BiLSTM AUC: {auc:.4f}")

except Exception as e:
    print(f"  BiLSTM AUC failed: {e}")


print("\nComputing Random DistilBERT AUC...")
# Note: we will try to load from the saved checkpoint folder for the random DistilBERT, but if that fails (e.g. if the checkpoint is missing or corrupted), we will fall back to loading from the main model dir which should at least have the final model weights, even if not the best checkpoint. This way we have a better chance of getting an AUC score for the report, even if it's not from the absolute best epoch.
try:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification

    config = DistilBertConfig(
        vocab_size=30522,
        n_layers=6, n_heads=12, dim=768, hidden_dim=3072,
        num_labels=NUM_LABELS,
        id2label=splits["id2label"],
        label2id=splits["label2id"],
    )

    # Load from saved checkpoint
    model_dir = "results/random_distilbert_model"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Find the best checkpoint folder
    import os
    checkpoints = [d for d in os.listdir(model_dir)
                   if d.startswith("checkpoint")]
    if checkpoints:
        # Sort by step number and take last
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
        best_dir = os.path.join(model_dir, checkpoints[-1])
    else:
        best_dir = model_dir

    print(f"  Loading from: {best_dir}")
    rand_model = DistilBertForSequenceClassification.from_pretrained(
        best_dir, ignore_mismatched_sizes=True
    ).to(DEVICE)
    rand_model.eval()

    all_probs = []
    batch_size = 32
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].tolist()
        enc   = tokenizer(batch, truncation=True, max_length=128,
                          padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = rand_model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)

    probs_arr = np.array(all_probs)
    auc = roc_auc_score(y_bin, probs_arr, average="macro", multi_class="ovr")
    all_results["DistilBERT (random)"]["auc"] = round(auc, 4)
    print(f"  Random DistilBERT AUC: {auc:.4f}")

except Exception as e:
    print(f"  Random DistilBERT AUC failed: {e}")

# Update all_results.json with new AUC scores
with open("results/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\n Updated AUC Results:")
for model, r in all_results.items():
    auc = r.get("auc", "N/A")
    print(f"  {model:<25} AUC: {auc}")

print("\n Done. results/all_results.json updated.")
