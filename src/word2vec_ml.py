"""
Word2Vec + ML classifiers for sentiment analysis.
Trains Word2Vec embeddings, then Logistic Regression and Random Forest.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "train.csv")
OUTPUT_FIG_DIR = os.path.join(SCRIPT_DIR, "outputs", "figures")
OUTPUT_MET_DIR = os.path.join(SCRIPT_DIR, "outputs", "metrics")

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_MET_DIR, exist_ok=True)

# --- 1. Load data ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")

# --- 2. Preprocess ---
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)  # remove mentions
    tokens = text.split()
    return tokens

print("Preprocessing...")
df["tokens"] = df["text"].apply(preprocess)

# --- 3. Train Word2Vec ---
print("Training Word2Vec...")
w2v_model = Word2Vec(
    sentences=df["tokens"].tolist(),
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    workers=4,
    seed=42,
)
print(f"  Vocabulary size: {len(w2v_model.wv)}")

# --- 4. Document vectors (mean pooling) ---
def doc_vector(tokens, model):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    if vecs:
        return np.mean(vecs, axis=0)
    return np.zeros(model.vector_size)

print("Creating document vectors...")
X = np.array([doc_vector(tokens, w2v_model) for tokens in df["tokens"]])
y = df["label"].values

# --- 5. Stratified split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# --- 6. Logistic Regression ---
print("\nTraining Logistic Regression...")
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

label_names = ["Bearish", "Bullish", "Neutral"]
print("\n=== Logistic Regression ===")
report_lr = classification_report(y_val, y_pred_lr, target_names=label_names)
print(report_lr)

# --- 7. Random Forest ---
print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("\n=== Random Forest ===")
report_rf = classification_report(y_val, y_pred_rf, target_names=label_names)
print(report_rf)

# --- 9. Confusion matrices ---
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_lr, display_labels=label_names, ax=ax, cmap="Blues")
ax.set_title("Word2Vec + Logistic Regression")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "cm_w2v_logreg.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_val, y_pred_rf, display_labels=label_names, ax=ax, cmap="Greens")
ax.set_title("Word2Vec + Random Forest")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FIG_DIR, "cm_w2v_rf.png"), dpi=150)
plt.close()

print(f"\nConfusion matrices saved to {OUTPUT_FIG_DIR}")

# --- 10. Save metrics JSON ---
metrics = {
    "logistic_regression": classification_report(y_val, y_pred_lr, target_names=label_names, output_dict=True),
    "random_forest": classification_report(y_val, y_pred_rf, target_names=label_names, output_dict=True),
}
metrics_path = os.path.join(OUTPUT_MET_DIR, "word2vec_ml_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Metrics saved to {metrics_path}")
print("\nDone!")
