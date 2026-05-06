"""
6_random_distilbert.py
Trains a DistilBERT with RANDOMLY INITIALISED weights (no pre-training).
This is the control experiment requested by my supervisor.

Purpose is to prove that pre-trained representations are what make
transformers powerful not just the architecture itself.

Expected result: much worse than pre-trained DistilBERT (86.7% F1),
probably similar to or worse than the TF-IDF baseline.
"""

import os, json, pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

def main():
    os.makedirs("results/figures", exist_ok=True)

    print("Loading splits...")
    with open("data/splits.pkl", "rb") as f:
        splits = pickle.load(f)

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val,   y_val   = splits["X_val"],   splits["y_val"]
    X_test,  y_test  = splits["X_test"],  splits["y_test"]
    id2label         = splits["id2label"]
    label2id         = splits["label2id"]
    LABELS           = [id2label[i] for i in range(len(id2label))]
    NUM_LABELS       = len(LABELS)

    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    #  Dataset preparation: convert to Hugging Face Datasets format for easier tokenisation
    raw = DatasetDict({
        "train": Dataset.from_dict({"text": X_train.tolist(), "label": y_train.tolist()}),
        "val":   Dataset.from_dict({"text": X_val.tolist(),   "label": y_val.tolist()}),
        "test":  Dataset.from_dict({"text": X_test.tolist(),  "label": y_test.tolist()}),
    })

    # Use DistilBERT tokeniser (vocabulary is still needed even with random weights)
    MODEL_ID  = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenise(batch):
        return tokenizer(batch["text"], truncation=True,
                         max_length=128, padding=False)

    tokenised     = raw.map(tokenise, batched=True, num_proc=None,
                             remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Build config with random weights (no pre-training loaded)
    print("\nBuilding randomly initialised DistilBERT (No pre-trained weights)...")
    config = DistilBertConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
        sinusoidal_pos_embds=False,
        n_layers=6,
        n_heads=12,
        dim=768,
        hidden_dim=3072,
        dropout=0.1,
        attention_dropout=0.1,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    # Note: DistilBERT has ~66M parameters, all randomly initialised here, so this is a much harder training task than the BiLSTM with 300k params. 
    model = DistilBertForSequenceClassification(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} (all randomly initialised)")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    out_dir = "results/random_distilbert_model"
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenised["train"],
        eval_dataset=tokenised["val"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nTraining randomly initialised DistilBERT...")
    train_result = trainer.train()
    print(f"Runtime: {train_result.metrics['train_runtime']:.0f}s")

    # Evaluation on test set
    print("\nEvaluating on test set...")
    pred_output = trainer.predict(tokenised["test"])
    logits      = pred_output.predictions
    test_preds  = np.argmax(logits, axis=1)

    test_acc  = accuracy_score(y_test, test_preds)
    test_f1   = f1_score(y_test, test_preds, average="macro")
    test_prec = precision_score(y_test, test_preds, average="macro")
    test_rec  = recall_score(y_test, test_preds, average="macro")

    print(f"\nRandom DistilBERT → Accuracy: {test_acc:.4f}  F1: {test_f1:.4f}")
    print(classification_report(y_test, test_preds, target_names=LABELS))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, test_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=LABELS, yticklabels=LABELS)
    ax.set_title("Confusion Matrix — Random DistilBERT (no pre-training)",
                 fontweight="bold")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("results/figures/random_distilbert_cm.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: results/figures/random_distilbert_cm.png")

    # Update the per-class F1 heatmap data to include random DistilBERT
    results = {}
    try:
        with open("results/all_results.json") as f:
            results = json.load(f)
    except FileNotFoundError:
        pass

    results["DistilBERT (random)"] = {
        "accuracy":  round(test_acc,  4),
        "f1_macro":  round(test_f1,   4),
        "precision": round(test_prec, 4),
        "recall":    round(test_rec,  4),
    }
# Update the per-class F1 scores for the heatmap (these are calculated from the confusion matrix)
    with open("results/all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Random DistilBERT complete.")
    print(f"\nKey comparison:")
    print(f"  Pre-trained DistilBERT : F1 = 0.8673")
    print(f"  Random DistilBERT      : F1 = {test_f1:.4f}")
    print(f"  Difference             : {0.8673 - test_f1:+.4f} percentage points")
    print(f"\nThis gap quantifies the contribution of pre-training.")

if __name__ == "__main__":
    main()
