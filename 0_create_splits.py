"""
One-time script to create data/splits.pkl needed by
2_baseline_model.py and 3_lstm_model.py.
The dataset is loaded from HuggingFace Datasets, cleaned, and split into train/val/test sets. 
The splits are saved as a pickle file for later use.
"""
import os, pickle, re
import numpy as np
from datasets import load_dataset

os.makedirs("data", exist_ok=True)

LABELS   = ["Normal", "Anxiety", "Depression", "Suicidal"]
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Loading dataset...")
ds = load_dataset(
    "ourafla/Mental-Health_Text-Classification_Dataset",
    data_files={
        "train": "mental_heath_unbanlanced.csv",
        "test":  "mental_health_combined_test.csv",
    }
)

def map_labels(example):
    example["label"] = label2id[example["status"]]
    return example

train_ds = ds["train"].map(map_labels, num_proc=None)
test_ds  = ds["test"].map(map_labels,  num_proc=None)

split    = train_ds.train_test_split(test_size=0.15, seed=42)
train_ds = split["train"]
val_ds   = split["test"]

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

X_train = np.array([clean_text(t) for t in train_ds["text"]])
X_val   = np.array([clean_text(t) for t in val_ds["text"]])
X_test  = np.array([clean_text(t) for t in test_ds["text"]])
y_train = np.array(train_ds["label"])
y_val   = np.array(val_ds["label"])
y_test  = np.array(test_ds["label"])

splits = {
    "X_train": X_train, "X_val": X_val, "X_test": X_test,
    "y_train": y_train, "y_val": y_val, "y_test": y_test,
    "label2id": label2id, "id2label": id2label,
}

with open("data/splits.pkl", "wb") as f:
    pickle.dump(splits, f)

print("Saved data/splits.pkl")
print(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
