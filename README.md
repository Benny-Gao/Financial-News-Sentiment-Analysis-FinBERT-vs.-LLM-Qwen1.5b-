# Financial News Sentiment Analysis: FinBERT vs. LLM (Qwen2.5-1.5B)

This is a basic sentiment analysis project using a Kaggle dataset. The project aims to compare the performance of an LLM and FinBERT in financial sentiment classification.

Kaggle project: https://www.kaggle.com/code/bennygao2/sentiment-analysis

An **informal** report is attached as a PDF.

The original project file is also attached.

## Results
| Model | Accuracy | Macro F1 | Weighted F1 | Negative recall |
|---|---|---|---|---|
| Always-neutral baseline | 59.4% | 0.25 | 0.44 | 0.00 |
| TF-IDF + Logistic Regression | 74.95% | 0.67 | 0.73 | 0.46 |
| Qwen2.5-1.5B-Instruct (zero-shot) | 74.95% | 0.72 | 0.75 | 0.57 |
| **FinBERT** (`ProsusAI/finbert`, as released) | **87.22%** | **0.86** | **0.87** | **0.97** |

<img width="1342" height="651" alt="results_overall" src="https://github.com/user-attachments/assets/8ccce89e-932d-4ef2-802a-1b2718aee519" />
