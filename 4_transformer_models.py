"""
4_transformer_models.py
Main training script for the three transformer models (BERT, DistilBERT, RoBERTa)
on the 4-class mental health text dataset.

I chose these three models because they are the most commonly used in recent NLP papers
and I wanted a good balance between performance and training speed on my RTX 3060.

This script also includes the Windows fixes I needed earlier.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# GLOBAL DEFINITIONS 
LABELS = ["Normal", "Anxiety", "Depression", "Suicidal"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

def map_labels(example):
    example["label"] = label2id[example["status"]]
    return example

# MAIN
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)

    print("CUDA available:", torch.cuda.is_available()) #Check if GPU is available
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # CONFIG
    # I decided to train these three models for fair comparison 
    MODEL_NAMES = ["bert-base-uncased", "distilbert-base-uncased", "roberta-base"]
    NUM_LABELS = 4
    MAX_LENGTH = 128
    BATCH_SIZE = 16 # Works well on my 3060
    EPOCHS = 3
    OUTPUT_DIR = "./mental_health_models"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # LOAD DATA 
    print("Loading dataset...")
    ds = load_dataset("ourafla/Mental-Health_Text-Classification_Dataset",
                      data_files={
                          "train": "mental_heath_unbanlanced.csv",
                          "test": "mental_health_combined_test.csv"
                      })

    train_ds = ds["train"]
    test_ds = ds["test"]

    print("Mapping labels...")
    train_ds = train_ds.map(map_labels, num_proc=1)
    test_ds = test_ds.map(map_labels, num_proc=1)
    # 10% validation split
    train_valid = train_ds.train_test_split(test_size=0.1, seed=42)
    train_ds = train_valid["train"]
    val_ds = train_valid["test"]

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # COMPUTE METRICS 
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    results = {}

    # TRAINING LOOP 
    for model_name in MODEL_NAMES:
        print(f"\n=== Training {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

        
        # num_proc=0 because multiprocessing kept crashing on Windows

        tokenized_train = train_ds.map(tokenize_function, batched=True, num_proc=0)
        tokenized_val   = val_ds.map(tokenize_function, batched=True, num_proc=0)
        tokenized_test  = test_ds.map(tokenize_function, batched=True, num_proc=0)

        tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])
        tokenized_test.set_format("torch", columns=["input_ids", "attention_mask", "label"])

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=NUM_LABELS, id2label=id2label, label2id=label2id
        )

        training_args = TrainingArguments(
            output_dir=f"{OUTPUT_DIR}/{model_name}",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            report_to="none",
            fp16 = True,                   
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trainer.train()

     
        
        # Final evaluation on test set

        test_results = trainer.evaluate(tokenized_test)
        predictions = trainer.predict(tokenized_test)
        preds = np.argmax(predictions.predictions, axis=-1)

        report = classification_report(test_ds["label"], preds, target_names=LABELS, output_dict=True)
        cm = confusion_matrix(test_ds["label"], preds)

        results[model_name] = {
            "test_accuracy": test_results["eval_accuracy"],
            "test_f1_macro": test_results["eval_f1_macro"],
            "classification_report": report,
            "confusion_matrix": cm
        }

        
         # Save model and tokenizer

        trainer.save_model(f"{OUTPUT_DIR}/{model_name}_final")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/{model_name}_final")
        
         # Save confusion matrix plot

        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
        plt.title(f"Confusion Matrix - {model_name}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig(f"{OUTPUT_DIR}/{model_name}_cm.png")
        plt.close()

    # FINAL RESULTS 
    comparison = pd.DataFrame({
        model: [results[model]["test_accuracy"], results[model]["test_f1_macro"]]
        for model in MODEL_NAMES
    }, index=["Accuracy", "Macro F1"]).T

    print("\n MODEL COMPARISON \n ", comparison)
    comparison.to_csv(f"{OUTPUT_DIR}/model_comparison.csv")
    print(f"\nDone. Results saved in folder: {OUTPUT_DIR}")

    best_model = max(results, key=lambda m: results[m]["test_f1_macro"])
    print(f"Best model: {best_model} with Macro F1 = {results[best_model]['test_f1_macro']:.4f}")