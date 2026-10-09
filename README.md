# Financial News Sentiment Analysis: FinBERT vs. LLM (Qwen2.5-1.5B)

This is a basic sentiment analysis project using a Kaggle dataset. The project aims to compare the performance of an LLM and FinBERT in financial sentiment classification.

Kaggle project: https://www.kaggle.com/code/bennygao2/sentiment-analysis

An **informal** report is attached as a PDF.

The original project file is also attached.

## Results

**Task:** 3-class sentiment classification (positive / neutral / negative) on 4,846 financial news sentences ([Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news), 50% agreement).
**Split:** stratified 80/20, `random_state=42` → 970 test sentences (576 neutral / 273 positive / 121 negative).

| Model | Accuracy | Macro F1 | Weighted F1 | Negative recall |
|---|---|---|---|---|
| Always-neutral baseline | 59.4% | 0.25 | 0.44 | 0.00 |
| TF-IDF + Logistic Regression | 74.95% | 0.67 | 0.73 | 0.46 |
| Qwen2.5-1.5B-Instruct (zero-shot) | 74.95% | 0.72 | 0.75 | 0.57 |
| **FinBERT** (`ProsusAI/finbert`, as released) | **87.22%** | **0.86** | **0.87** | **0.97** |

![Overall performance](assets/results_overall.png)
![Recall by class](assets/results_recall.png)

### Key findings

- **FinBERT is best overall**, especially on the minority negative class (recall 0.97 vs 0.46 and 0.57).
- **FinBERT over-interprets neutral events**: 42 of the 43 sentences where only FinBERT failed are neutral.
- **TF-IDF leans on the majority class** (neutral recall 0.92, but negative 0.46 and positive 0.52).
- **Qwen is cautious on negatives** (precision 0.93, recall 0.57) and often reads neutral business news as positive.
- **Overlap:** all three models are correct on 545/970 sentences and all wrong on 24. Only FinBERT is correct on 80, only Qwen on 34, only TF-IDF on 23.

### Setup

- TF-IDF: `TfidfVectorizer()` + `LogisticRegression(max_iter=1000)`, default settings, fitted on the training set only.
- FinBERT: used as released, no fine-tuning.
- Qwen2.5-1.5B-Instruct: zero-shot prompt, greedy decoding.

> **Caveat:** `ProsusAI/finbert` was fine-tuned on the Financial PhraseBank, the corpus this dataset comes from, and the test split is random. Some test sentences may have been seen during its training, so its score may be optimistic.
> 
<img width="1342" height="651" alt="results_overall" src="https://github.com/user-attachments/assets/8ccce89e-932d-4ef2-802a-1b2718aee519" />
