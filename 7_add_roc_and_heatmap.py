"""
7_add_roc_and_heatmap.py

Adds two things requested by my supervisor:
  1. ROC AUC scores (macro OvR) for all models that support it
  2. Per-class F1 heatmap across all models
  3. Updated final comparison table with AUC column

"""

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

os.makedirs("results/figures", exist_ok=True)

with open("data/splits.pkl", "rb") as f:
    splits = pickle.load(f)
X_test   = splits["X_test"]
y_test   = splits["y_test"]
id2label = splits["id2label"]
LABELS   = [id2label[i] for i in range(len(id2label))]
NUM_LABELS = len(LABELS)

# Binarize labels for OvR AUC
y_bin = label_binarize(y_test, classes=list(range(NUM_LABELS)))

# ── 1. TF-IDF AUC ─────────────────────────────────────────────────────────────
print("Computing TF-IDF AUC...")
auc_results = {}
try:
    with open("results/baseline_pipeline.pkl", "rb") as f:
        baseline = pickle.load(f)
    probs = baseline.predict_proba(X_test)
    auc   = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
    auc_results["TF-IDF + LR"] = round(auc, 4)
    print(f"  TF-IDF AUC: {auc:.4f}")
except Exception as e:
    print(f"  TF-IDF AUC failed: {e}")

# ── 2. Transformer AUC ────────────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORMER_CONFIGS = [
    ("BERT",               "bert-base-uncased",        "results/bert_model"),
    ("DistilBERT",         "distilbert-base-uncased",  "results/distilbert_model"),
    ("RoBERTa",            "roberta-base",             "results/roberta_model"),
    ("DistilBERT (random)","distilbert-base-uncased",  "results/random_distilbert_model"),
]

# Also try the mental_health_models folder from original pipeline
TRANSFORMER_CONFIGS_ALT = [
    ("BERT",       "bert-base-uncased",        "mental_health_models/bert-base-uncased_final"),
    ("DistilBERT", "distilbert-base-uncased",  "mental_health_models/distilbert-base-uncased_final"),
    ("RoBERTa",    "roberta-base",             "mental_health_models/roberta-base_final"),
]

def get_probs(model_id, model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model     = model.to(DEVICE); model.eval()
    all_probs = []
    batch_size = 32
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size].tolist()
        enc   = tokenizer(batch, truncation=True, max_length=128,
                          padding=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs)
    del model; torch.cuda.empty_cache()
    return np.array(all_probs)

for name, model_id, model_dir in TRANSFORMER_CONFIGS:
    if name in auc_results:
        continue
    # Try primary dir first, then alt
    dirs_to_try = [model_dir]
    for alt_name, alt_id, alt_dir in TRANSFORMER_CONFIGS_ALT:
        if alt_name == name:
            dirs_to_try.append(alt_dir)

    for d in dirs_to_try:
        if not os.path.exists(d):
            continue
        try:
            print(f"Computing {name} AUC from {d}...")
            probs = get_probs(model_id, d)
            auc   = roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
            auc_results[name] = round(auc, 4)
            print(f"  {name} AUC: {auc:.4f}")
            break
        except Exception as e:
            print(f"  {name} from {d} failed: {e}")

# ── 3. Load all results and merge AUC ─────────────────────────────────────────
with open("results/all_results.json") as f:
    all_results = json.load(f)

for model, auc in auc_results.items():
    if model in all_results:
        all_results[model]["auc"] = auc

with open("results/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nAUC scores added to all_results.json")

# ── 4. Per-class F1 heatmap ───────────────────────────────────────────────────
print("\nGenerating per-class F1 heatmap...")

# Hardcode per-class F1 from confusion matrices we have
# Derived from confusion matrix values
per_class_f1 = {
    "TF-IDF + LR": {
        "Normal":     round(2*206/(2*206+17+11+19), 3),
        "Anxiety":    round(2*202/(2*202+28+7+32), 3),
        "Depression": round(2*159/(2*159+40+17+32), 3),
        "Suicidal":   round(2*202/(2*202+29+13+40), 3),
    },
    "BiLSTM": {
        "Normal":     round(2*219/(2*219+7+22+17), 3),
        "Anxiety":    round(2*178/(2*178+33+15+33), 3),
        "Depression": round(2*122/(2*122+70+23+33), 3),
        "Suicidal":   round(2*206/(2*206+29+10+70), 3),
    },
    "BERT": {
        "Normal":     round(2*235/(2*235+5+12+5), 3),
        "Anxiety":    round(2*199/(2*199+33+4+18), 3),
        "Depression": round(2*192/(2*192+23+15+18), 3),
        "Suicidal":   round(2*243/(2*243+5+0+23), 3),
    },
    "DistilBERT": {
        "Normal":     round(2*242/(2*242+1+14+5), 3),
        "Anxiety":    round(2*194/(2*194+39+1+21), 3),
        "Depression": round(2*191/(2*191+21+15+21), 3),
        "Suicidal":   round(2*235/(2*235+11+1+21), 3),
    },
    "RoBERTa": {
        "Normal":     round(2*241/(2*241+3+0+4), 3),
        "Anxiety":    round(2*210/(2*210+38+0+18), 3),
        "Depression": round(2*224/(2*224+2+4+18), 3),
        "Suicidal":   round(2*245/(2*245+3+0+2), 3),
    },
}

# Add random DistilBERT if available
if "DistilBERT (random)" in all_results:
    per_class_f1["DistilBERT (random)"] = {
        c: 0.0 for c in LABELS  # Will be 0 until CM is available
    }

df_heatmap = pd.DataFrame(per_class_f1).T
df_heatmap = df_heatmap[LABELS]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df_heatmap, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.4, vmax=1.0, ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 11})
ax.set_title("Per-Class F1 Score — All Models", fontsize=14, fontweight="bold")
ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Model", fontsize=12)
plt.tight_layout()
plt.savefig("results/figures/per_class_f1_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/per_class_f1_heatmap.png")

# ── 5. Updated summary table ──────────────────────────────────────────────────
ORDER = ["TF-IDF + LR", "BiLSTM", "DistilBERT (random)",
         "BERT", "DistilBERT", "RoBERTa"]
rows = []
for m in ORDER:
    if m not in all_results:
        continue
    r = all_results[m]
    rows.append({
        "Model":     m,
        "Accuracy":  r.get("accuracy", 0),
        "Macro F1":  r.get("f1_macro", 0),
        "Precision": r.get("precision", 0),
        "Recall":    r.get("recall", 0),
        "AUC (OvR)": r.get("auc", "N/A"),
    })

df = pd.DataFrame(rows).set_index("Model")
df.to_csv("results/final_comparison_table_v2.csv")

print("\n── Updated Results Table ──")
print(df.to_string())
print("\nSaved: results/final_comparison_table_v2.csv")
print("\n✓ All additional evaluation complete.")
