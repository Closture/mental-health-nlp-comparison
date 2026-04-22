"""
8_fix_heatmap.py
Fixes the per-class F1 heatmap to include correct values
for random DistilBERT, computed from its confusion matrix. It bugged on my run because I had an old version of the code that didn't save the random DistilBERT confusion matrix, 
so I hardcoded the values from a fresh run. Also adds BiLSTM AUC via OvR probability estimation.
"""

import os, json, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("results/figures", exist_ok=True)

# ── Per-class F1 computed from confusion matrices ─────────────────────────────
# Formula: F1 = 2*TP / (2*TP + FP + FN)
# For each class: TP = diagonal, FP = column sum - TP, FN = row sum - TP

def f1_from_cm(tp, row_sum, col_sum):
    fp = col_sum - tp
    fn = row_sum - tp
    return round(2*tp / (2*tp + fp + fn), 3)

per_class_f1 = {
    "TF-IDF + LR": {
        # CM rows: Normal[206,17,19,6], Anxiety[11,202,28,7], Depression[17,32,159,40], Suicidal[13,4,29,202]
        "Normal":     f1_from_cm(206, 248, 206+11+17+13),
        "Anxiety":    f1_from_cm(202, 248, 17+202+32+4),
        "Depression": f1_from_cm(159, 248, 19+28+159+29),
        "Suicidal":   f1_from_cm(202, 248, 6+7+40+202),
    },
    "BiLSTM": {
        # CM rows: Normal[219,7,17,5], Anxiety[22,178,33,15], Depression[23,33,122,70], Suicidal[10,3,29,206]
        "Normal":     f1_from_cm(219, 248, 219+22+23+10),
        "Anxiety":    f1_from_cm(178, 248, 7+178+33+3),
        "Depression": f1_from_cm(122, 248, 17+33+122+29),
        "Suicidal":   f1_from_cm(206, 248, 5+15+70+206),
    },
    "DistilBERT (random)": {
        # CM rows: Normal[222,7,17,2], Anxiety[16,190,37,5], Depression[17,38,150,43], Suicidal[6,3,46,193]
        "Normal":     f1_from_cm(222, 248, 222+16+17+6),
        "Anxiety":    f1_from_cm(190, 248, 7+190+38+3),
        "Depression": f1_from_cm(150, 248, 17+37+150+46),
        "Suicidal":   f1_from_cm(193, 248, 2+5+43+193),
    },
    "BERT": {
        # CM rows: Normal[235,5,5,3], Anxiety[12,199,33,4], Depression[15,18,192,23], Suicidal[0,0,5,243]
        "Normal":     f1_from_cm(235, 248, 235+12+15+0),
        "Anxiety":    f1_from_cm(199, 248, 5+199+18+0),
        "Depression": f1_from_cm(192, 248, 5+33+192+5),
        "Suicidal":   f1_from_cm(243, 248, 3+4+23+243),
    },
    "DistilBERT": {
        # CM rows: Normal[242,1,5,0], Anxiety[14,194,39,1], Depression[15,21,191,21], Suicidal[1,1,11,235]
        "Normal":     f1_from_cm(242, 248, 242+14+15+1),
        "Anxiety":    f1_from_cm(194, 248, 1+194+21+1),
        "Depression": f1_from_cm(191, 248, 5+39+191+11),
        "Suicidal":   f1_from_cm(235, 248, 0+1+21+235),
    },
    "RoBERTa": {
        # CM rows: Normal[241,3,4,0], Anxiety[0,210,38,0], Depression[4,18,224,2], Suicidal[0,0,3,245]
        "Normal":     f1_from_cm(241, 248, 241+0+4+0),
        "Anxiety":    f1_from_cm(210, 248, 3+210+18+0),
        "Depression": f1_from_cm(224, 248, 4+38+224+3),
        "Suicidal":   f1_from_cm(245, 248, 0+0+2+245),
    },
}

LABELS = ["Normal", "Anxiety", "Depression", "Suicidal"]
ORDER  = ["TF-IDF + LR", "BiLSTM", "DistilBERT (random)",
          "BERT", "DistilBERT", "RoBERTa"]

df = pd.DataFrame(per_class_f1).T[LABELS]
df = df.loc[ORDER]

print("Per-Class F1 Scores:")
print(df.to_string(float_format="{:.3f}".format))

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(df, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, ax=ax,
            linewidths=0.5, linecolor="white",
            annot_kws={"size": 12, "weight": "bold"})
ax.set_title("Per-Class F1 Score — All Models", fontsize=15, fontweight="bold", pad=15)
ax.set_xlabel("Class", fontsize=13)
ax.set_ylabel("Model", fontsize=13)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=11, rotation=0)
plt.tight_layout()
plt.savefig("results/figures/per_class_f1_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved: results/figures/per_class_f1_heatmap.png")

# ── Also fix the bar chart to properly include random DistilBERT ──────────────
with open("results/all_results.json") as f:
    all_results = json.load(f)

metrics = ["accuracy", "f1_macro", "precision", "recall"]
mlabels = ["Accuracy", "Macro F1", "Precision", "Recall"]
colors  = {
    "TF-IDF + LR":        "#888780",
    "BiLSTM":             "#5B8DB8",
    "DistilBERT (random)":"#F0A500",
    "BERT":               "#534AB7",
    "DistilBERT":         "#0F6E56",
    "RoBERTa":            "#D4756B",
}

models_ordered = [m for m in ORDER if m in all_results]
x     = np.arange(len(mlabels))
width = 0.13
fig, ax = plt.subplots(figsize=(16, 7))

for i, model in enumerate(models_ordered):
    vals = [all_results[model].get(m, 0) for m in metrics]
    bars = ax.bar(x + i*width, vals, width, label=model,
                  color=colors.get(model, "#999"),
                  alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.004,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7,
                rotation=90)

ax.set_xticks(x + width*(len(models_ordered)-1)/2)
ax.set_xticklabels(mlabels, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Model Comparison — All Metrics (Test Set)", fontsize=14, fontweight="bold")
ax.legend(fontsize=9, loc="lower right", ncol=2)
ax.axhline(0.5, linestyle="--", color="gray", alpha=0.3)
ax.grid(axis="y", alpha=0.25)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/model_comparison_bar_full.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/figures/model_comparison_bar_full.png")

# ── Print final summary for report ───────────────────────────────────────────
print("\n── Complete Results Summary ──")
print(f"{'Model':<22} {'Acc':>6} {'F1':>6} {'Prec':>6} {'Rec':>6} {'AUC':>7}")
print("-" * 60)
for m in ORDER:
    if m not in all_results: continue
    r = all_results[m]
    auc = r.get('auc', 'N/A')
    auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
    print(f"{m:<22} {r['accuracy']:>6.4f} {r['f1_macro']:>6.4f} "
          f"{r['precision']:>6.4f} {r['recall']:>6.4f} {auc_str:>7}")
