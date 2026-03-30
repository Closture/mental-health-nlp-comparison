# Mental Health Text Classification: Comparative Study

**Final Year Project**  
School of Electronic Engineering and Computer Science  
(2025/26)

This project compares classical machine learning and modern transformer models for detecting four mental health states (Normal, Anxiety, Depression, Suicidal) from user-generated text.

### Final Results

| Model                        | Accuracy | Macro F1   | Rank |
|------------------------------|----------|------------|------|
| **RoBERTa-base**             | 92.74%   | **92.77%** | 1    |
| BERT-base                    | 87.60%   | 87.41%     | 2    |
| DistilBERT-base              | 86.90%   | 86.73%     | 3    |
| TF-IDF + Logistic Regression | 77.52%   | 77.43%     | 4    |
| BiLSTM (with attention)      | 73.08%   | 72.47%     | 5    |

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

