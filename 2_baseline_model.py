"""
2_baseline_model.py
TF-IDF + Logistic Regression baseline (4-class version).
I added this as the classical baseline so I could compare it against the transformers.
"""

import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

os.makedirs("results/figures", exist_ok=True)

# Load splits
print("Loading data splits...")
with open("data/splits.pkl", "rb") as f:
    splits = pickle.load(f)

X_train, y_train = splits["X_train"], splits["y_train"]
X_val,   y_val   = splits["X_val"],   splits["y_val"]
X_test,  y_test  = splits["X_test"],  splits["y_test"]
id2label         = splits["id2label"]
LABELS           = [id2label[i] for i in range(len(id2label))]

print(f"Classes: {LABELS}")

# Training the baseline model
print("Training TF-IDF + Logistic Regression...")
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )),
    ("clf", LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced", 
        random_state=42,
        n_jobs=-1,
    )),
])

pipeline.fit(X_train, y_train)
print("Training complete.")

# Validation 
val_preds = pipeline.predict(X_val)
val_acc   = accuracy_score(y_val, val_preds)
val_f1    = f1_score(y_val, val_preds, average="macro")
print(f"\nValidation — Accuracy: {val_acc:.4f}  |  Macro F1: {val_f1:.4f}")

# Test evaluation 
test_preds = pipeline.predict(X_test)
test_acc   = accuracy_score(y_test, test_preds)
test_f1    = f1_score(y_test, test_preds, average="macro")
test_prec  = precision_score(y_test, test_preds, average="macro")
test_rec   = recall_score(y_test, test_preds, average="macro")

print(f"Test  — Accuracy: {test_acc:.4f}  |  Macro F1: {test_f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, test_preds, target_names=LABELS))

# Confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=LABELS, yticklabels=LABELS)
ax.set_title("Confusion Matrix — TF-IDF + Logistic Regression", fontweight="bold")
ax.set_ylabel("True label")
ax.set_xlabel("Predicted label")
plt.tight_layout()
plt.savefig("results/figures/baseline_cm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/baseline_cm.png")

# Top features per class (I thought this would be useful for the discussion section)
tfidf         = pipeline.named_steps["tfidf"]
clf           = pipeline.named_steps["clf"]
feature_names = np.array(tfidf.get_feature_names_out())
top_n         = 15

fig, axes = plt.subplots(1, 4, figsize=(22, 6))
for i, (label, ax) in enumerate(zip(LABELS, axes)):
    coefs   = clf.coef_[i]
    top_idx = np.argsort(coefs)[-top_n:][::-1]
    ax.barh(feature_names[top_idx][::-1], coefs[top_idx][::-1], color="#5B8DB8")
    ax.set_title(f"Top words → {label}", fontweight="bold", fontsize=11)
    ax.set_xlabel("Coefficient")
plt.suptitle("TF-IDF Feature Importance by Class", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig("results/figures/baseline_features.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/baseline_features.png")

# Save results for the final comparison
results = {}
try:
    with open("results/all_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    pass

results["TF-IDF + LR"] = {
    "accuracy":  round(test_acc,  4),
    "f1_macro":  round(test_f1,   4),
    "precision": round(test_prec, 4),
    "recall":    round(test_rec,  4),
}

with open("results/all_results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("results/baseline_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n✓ Baseline complete. Results saved to results/all_results.json\n")
