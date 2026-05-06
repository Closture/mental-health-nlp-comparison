# Mental Health Text Classification — Comparative NLP Study
## QMUL Final Year Project 2025/26 | Chizara Vincent Okoye Udeh

### Project Overview
A systematic comparative study of six machine learning models for 
multi-class mental health classification from social media text.
Models evaluated span three generations of NLP methodology.

### Models Evaluated
| Model | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| TF-IDF + Logistic Regression | 77.52% | 77.43% | 0.928 |
| BiLSTM with Attention | 73.08% | 72.47% | 0.912 |
| DistilBERT (random init) | 76.11% | 76.07% | 0.935 |
| BERT | 87.60% | 87.41% | 0.967 |
| DistilBERT | 86.90% | 86.73% | 0.939 |
| **RoBERTa** | **92.74%** | **92.77%** | **0.965** |

### Dataset
- Source: ourafla/Mental-Health_Text-Classification_Dataset (HuggingFace)
- 50,604 Reddit posts across 4 classes: Normal, Anxiety, Depression, Suicidal
- Split: 42,170 train / 7,442 val / 992 test

### Key Findings
- RoBERTa outperforms the classical baseline by 15.34 percentage points
- Ablation study: pre-training contributes 10.66 F1 points to DistilBERT
- Cross-validation confirms TF-IDF (77.89% ± 0.18%) and BiLSTM (77.15% ± 0.62%)
- Depression-Suicidal boundary is the most persistent classification challenge

### Run Order
```bash
pip install -r requirements.txt
python 0_create_splits.py
python 2_baseline_model.py
python 3_lstm_model.py
python 6_random_distilbert.py
python 7_add_roc_and_heatmap.py
python 8_fix_heatmap.py
python 9_cross_validation.py
python 10_compute_missing_auc.py
python 5_evaluation.py
```

### Environment
- Python 3.11, PyTorch 2.5.1, CUDA 12.1
- NVIDIA RTX 3060 Laptop GPU
- HuggingFace Transformers 5.4.0

### Supervisor
Dr Jinhua Liang — Queen Mary University of London