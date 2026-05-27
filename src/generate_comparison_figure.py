"""Generate a grouped bar chart comparing F1-scores (macro) across all models."""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Load metrics
with open("outputs/metrics/traditional_ml_metrics.json") as f:
    trad = json.load(f)
with open("outputs/metrics/word2vec_ml_metrics.json") as f:
    w2v = json.load(f)

# Extract macro F1 scores
models = [
    "TF-IDF + LogReg",
    "TF-IDF + Random Forest",
    "W2V + LogReg",
    "W2V + Random Forest",
    "BERT-base",
    "FinBERT",
    "DistilBERT",
]

f1_scores = [
    trad["logistic_regression"]["classification_report"]["macro avg"]["f1-score"],
    trad["random_forest"]["classification_report"]["macro avg"]["f1-score"],
    w2v["logistic_regression"]["macro avg"]["f1-score"],
    w2v["random_forest"]["macro avg"]["f1-score"],
    0.77,  # BERT-base hardcoded
    0.83,  # FinBERT hardcoded
    0.75,  # DistilBERT hardcoded
]

accuracies = [
    trad["logistic_regression"]["accuracy"],
    trad["random_forest"]["accuracy"],
    w2v["logistic_regression"]["accuracy"],
    w2v["random_forest"]["accuracy"],
    0.84,
    0.8685,
    0.82,
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, f1_scores, width, label='Macro F1', color='steelblue')
bars2 = ax.bar(x + width/2, accuracies, width, label='Accuracy', color='coral')

ax.set_ylabel('Score')
ax.set_title('Model Comparison: Macro F1-Score and Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha='right')
ax.legend()
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
os.makedirs("outputs/figures", exist_ok=True)
plt.savefig("outputs/figures/model_comparison_f1.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved: outputs/figures/model_comparison_f1.png\n")

# Summary table
df = pd.DataFrame({
    "Model": models,
    "Macro F1": [f"{x:.4f}" for x in f1_scores],
    "Accuracy": [f"{x:.4f}" for x in accuracies],
})
print(df.to_string(index=False))
