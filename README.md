# Mental Health Text Classification: Comparative Study

**Final Year Project**  
School of Electronic Engineering and Computer Science  
(2025/26)

This project compares classical machine learning and modern transformer models for detecting four mental health states (Normal, Anxiety, Depression, Suicidal) from user-generated text.

### Final Results

RoBERTa-base: Accuracy: 92.74%; Macro F1: **92.77%**; Rank: 1
BERT-base: Accuracy: 87.60%; Macro F1: 87.41%; Rank: 2
DistilBERT-base: Accuracy: 86.90%; Macro F1: 86.73%; Rank: 3
TF-IDF + Logistic Regression: Accuracy: 77.52%; Macro F1: 77.43%; Rank: 4
BiLSTM (with attention): Accuracy: 73.08%; Macro F1: 72.47%; Rank: 5 

RoBERTa clearly outperformed the others.

All plots and the final table are saved in results/figures/ and results/final_comparison_table.csv


4_transformer_models.py – BERT, DistilBERT, RoBERTa training
3_lstm_model.py – BiLSTM with attention
2_baseline_model.py – TF-IDF baseline
5_evaluation.py – Generates all comparison charts and table


### How to Run

```bash
pip install -r requirements.txt
python 0_create_splits.py
python 2_baseline_model.py
python 3_lstm_model.py
python 4_transformer_models.py
python 5_evaluation.py

