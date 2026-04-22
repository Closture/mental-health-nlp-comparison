"""
One-time script to add the transformer results from my
mental_health_nlp_pipeline.py run into all_results.json as I was 
originally saving the transformer results separately in model_comparison.csv. 
Though this can be rerun by anyone who trains the transformers, 
I ran it once and then ran 5_evaluation.py to get the final comparison figures.
"""
import json, os

os.makedirs("results", exist_ok=True)

# Load existing results
results = {}
try:
    with open("results/all_results.json") as f:
        results = json.load(f)
except FileNotFoundError:
    pass

# Add transformer results from completed training run
# These come directly from the terminal output  I got when I ran mental_health_nlp_pipeline.py with the three transformers.
results["BERT"] = {
    "accuracy":  0.8760,
    "f1_macro":  0.8741,
    "precision": 0.8760,
    "recall":    0.8760,
}
results["DistilBERT"] = {
    "accuracy":  0.8690,
    "f1_macro":  0.8673,
    "precision": 0.8690,
    "recall":    0.8690,
}
results["RoBERTa"] = {
    "accuracy":  0.9274,
    "f1_macro":  0.9277,
    "precision": 0.9274,
    "recall":    0.9274,
}

with open("results/all_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Transformer results added to results/all_results.json")
print("\nAll models now in results:")
for model, r in results.items():
    print(f"  {model:15s} → Accuracy: {r['accuracy']:.4f}  F1: {r['f1_macro']:.4f}")

