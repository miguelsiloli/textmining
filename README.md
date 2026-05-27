# Text Mining: Financial Tweet Sentiment Classification

**Course:** Text Mining — Spring Semester 2025/2026  
**Deadline:** 15 June 2026

## Objective

Develop an NLP model to classify financial tweets as **Bearish (0)**, **Bullish (1)**, or **Neutral (2)**.

## Project Structure

```
src/
├── data/                   # Train (9,543 tweets) and Test (2,486 tweets)
├── preprocessing/          # Text normalization, entity masking
├── topics/                 # BERTopic topic modeling
├── notebooks/              # EDA, BERT training notebooks
├── models/                 # Model training scripts
├── outputs/                # Figures, metrics, predictions
├── traditional_ml.py       # TF-IDF + LogReg/RF pipeline
├── word2vec_ml.py          # Word2Vec + classifiers
├── train_advanced_finbert.py  # Extended pretraining + multi-task
report/
├── report_XX.tex           # LaTeX report
├── report_XX.pdf           # Compiled PDF (12 pages)
```

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| W2V + LogReg | 0.62 | 0.52 |
| TF-IDF + RF | 0.78 | 0.68 |
| TF-IDF + LogReg | 0.81 | 0.75 |
| BERT-base | 0.84 | 0.77 |
| **FinBERT** | **0.87** | **0.83** |

## How to Run

```bash
cd src/
python traditional_ml.py        # Traditional ML baselines
python word2vec_ml.py            # Word2Vec pipeline
python train_advanced_finbert.py # Advanced FinBERT (requires GPU)
```
