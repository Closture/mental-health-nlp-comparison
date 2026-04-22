"""
5_evaluation.py
This script combines all the model results (baseline, BiLSTM, and the three transformers)
and creates the final comparison plots and table for the report.

I run this after training everything so I have one clean place with all the numbers and figures.
"""

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)

# 1. Load results
print("Loading results...")
with open("results/all_results.json") as f:
    all_results = json.load(f)

# Also pull transformer results from model_comparison.csv if it exists
# (saved by the original mental_health_nlp_pipeline.py)
try:
    mc = pd.read_csv("mental_health_models/model_comparison.csv", index_col=0)
    name_map = {
        "bert-base-uncased":       "BERT",
        "distilbert-base-uncased": "DistilBERT",
        "roberta-base":            "RoBERTa",
    }
    for old_name, new_name in name_map.items():
        if old_name in mc.index and new_name not in all_results:
            all_results[new_name] = {
                "accuracy": round(float(mc.loc[old_name, "Accuracy"]), 4),
                "f1_macro": round(float(mc.loc[old_name, "Macro F1"]), 4),
                "precision": 0.0,
                "recall": 0.0,
            }
    print(f"  Loaded transformer results from model_comparison.csv")
except Exception as e:
    print(f"  Note: {e}")

print(f"Models found: {list(all_results.keys())}")

# Define display order
ORDER = ["TF-IDF + LR", "BiLSTM", "BERT", "DistilBERT", "RoBERTa"]
models = [m for m in ORDER if m in all_results]
# Add any extra models not in ORDER
models += [m for m in all_results if m not in models]

metrics  = ["accuracy", "f1_macro", "precision", "recall"]
mlabels  = ["Accuracy", "Macro F1", "Precision", "Recall"]
colors   = ["#888780", "#5B8DB8", "#534AB7", "#0F6E56", "#D4756B"]

# 2. Grouped bar chart 
print("Generating comparison bar chart...")
x     = np.arange(len(mlabels))
width = 0.15
fig, ax = plt.subplots(figsize=(14, 6))

for i, (model, color) in enumerate(zip(models, colors)):
    vals = [all_results[model].get(m, 0) for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=model,
                  color=color, alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(mlabels, fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison  All Metrics (Test Set)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="lower right")
ax.axhline(0.5, linestyle="--", color="gray", alpha=0.3)
ax.grid(axis="y", alpha=0.25, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/model_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/figures/model_comparison_bar.png")

# 3. Radar chart 
print("Generating radar chart...")
N      = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(mlabels, fontsize=11)
ax.set_ylim(0.4, 1.0)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(["0.5","0.6","0.7","0.8","0.9","1.0"], fontsize=8)
ax.grid(alpha=0.3)

for model, color in zip(models, colors):
    vals = [all_results[model].get(m, 0) for m in metrics]
    vals_closed = vals + [vals[0]]
    ax.plot(angles, vals_closed, color=color, linewidth=2, label=model)
    ax.fill(angles, vals_closed, color=color, alpha=0.07)

ax.set_title("Radar Chart — Model Comparison", fontsize=13,
             fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)
plt.tight_layout()
plt.savefig("results/figures/model_comparison_radar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/figures/model_comparison_radar.png")

# 4. F1 improvement line plot 
print("Generating F1 progression plot...")
fig, ax = plt.subplots(figsize=(10, 5))
f1_vals = [all_results[m].get("f1_macro", 0) for m in models]
acc_vals = [all_results[m].get("accuracy", 0) for m in models]

ax.plot(models, f1_vals,  "o-", color="#534AB7", lw=2, ms=8, label="Macro F1")
ax.plot(models, acc_vals, "s--", color="#D4756B", lw=2, ms=8, label="Accuracy")
for i, (m, f, a) in enumerate(zip(models, f1_vals, acc_vals)):
    ax.annotate(f"{f:.3f}", (i, f), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=9, color="#534AB7")
    ax.annotate(f"{a:.3f}", (i, a), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=9, color="#D4756B")

ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Performance Progression", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0.6, 1.0)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/model_progression.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results/figures/model_progression.png")

# 5. Final summary table 
print("\nGenerating summary table...")
rows = []
for m in models:
    r = all_results[m]
    rows.append({
        "Model":     m,
        "Accuracy":  r.get("accuracy", 0),
        "Macro F1":  r.get("f1_macro", 0),
        "Precision": r.get("precision", 0),
        "Recall":    r.get("recall", 0),
    })

df = pd.DataFrame(rows).set_index("Model")
df["Rank (F1)"] = df["Macro F1"].rank(ascending=False).astype(int)
df.to_csv("results/final_comparison_table.csv")

print("\n── Final Results Table ──")
print(df.to_string(float_format="{:.4f}".format))
print("\nSaved: results/final_comparison_table.csv")

print("\n✓ Evaluation complete!")
print("\nReport-ready figures saved to results/figures/:")
for f in sorted(os.listdir("results/figures")):
    print(f"  {f}")
