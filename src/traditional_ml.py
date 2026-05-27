"""
Traditional ML pipeline for sentiment classification.
Trains Logistic Regression and Random Forest on TF-IDF features.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.feature_selection import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Config ---
LABEL_MAP = {0: "Bearish", 1: "Bullish", 2: "Neutral"}
DATA_PATH = "data/train.csv"
OUTPUT_FIG_DIR = "outputs/figures"
OUTPUT_METRICS_DIR = "outputs/metrics"

# Create directories
os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)
os.makedirs(OUTPUT_METRICS_DIR, exist_ok=True)

# --- 1. Load data ---
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")
print(f"  Label distribution:\n{df['label'].value_counts().sort_index()}")

# --- 2. Preprocess ---
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'@\w+', '', text)               # @mentions
    text = re.sub(r'[^a-z0-9\s]', '', text)        # special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("Preprocessing text...")
df['clean_text'] = df['text'].apply(preprocess)

# --- 3. TF-IDF ---
print("Applying TF-IDF vectorization...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X = tfidf.fit_transform(df['clean_text'])
y = df['label'].values
print(f"  TF-IDF matrix shape: {X.shape}")

# --- 4. Train/Val split ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

# --- 5. Logistic Regression ---
print("\nTraining Logistic Regression...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_val)

print("\n=== Logistic Regression ===")
print(classification_report(y_val, y_pred_lr, target_names=list(LABEL_MAP.values())))
acc_lr = accuracy_score(y_val, y_pred_lr)
print(f"Accuracy: {acc_lr:.4f}")

# --- 6. Random Forest ---
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("\n=== Random Forest ===")
print(classification_report(y_val, y_pred_rf, target_names=list(LABEL_MAP.values())))
acc_rf = accuracy_score(y_val, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}")

# --- 9. Confusion matrices ---
def save_cm(y_true, y_pred, path, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(LABEL_MAP.values()),
                yticklabels=list(LABEL_MAP.values()), ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")

save_cm(y_val, y_pred_lr, os.path.join(OUTPUT_FIG_DIR, "cm_logreg.png"), "Logistic Regression - Confusion Matrix")
save_cm(y_val, y_pred_rf, os.path.join(OUTPUT_FIG_DIR, "cm_rf.png"), "Random Forest - Confusion Matrix")

# --- 10. Save metrics JSON ---
def get_report_dict(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=list(LABEL_MAP.values()), output_dict=True)

metrics = {
    "logistic_regression": {
        "accuracy": acc_lr,
        "classification_report": get_report_dict(y_val, y_pred_lr)
    },
    "random_forest": {
        "accuracy": acc_rf,
        "classification_report": get_report_dict(y_val, y_pred_rf)
    }
}

# --- 11. TF-IDF stats: vocab size + top 10 features per class ---
print("\nComputing TF-IDF stats (chi2)...")
vocab_size = len(tfidf.vocabulary_)
feature_names = np.array(tfidf.get_feature_names_out())

chi2_scores, _ = chi2(X_train, y_train)
top_features_per_class = {}
for label_id, label_name in LABEL_MAP.items():
    # For per-class ranking, compute chi2 with binary label
    mask = (y_train == label_id).astype(int)
    scores, _ = chi2(X_train, mask)
    top_idx = np.argsort(scores)[-10:][::-1]
    top_features_per_class[label_name] = feature_names[top_idx].tolist()
    print(f"  Top 10 for {label_name}: {top_features_per_class[label_name]}")

metrics["tfidf_stats"] = {
    "vocabulary_size": vocab_size,
    "top_10_features_per_class": top_features_per_class
}

metrics_path = os.path.join(OUTPUT_METRICS_DIR, "traditional_ml_metrics.json")
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {metrics_path}")
print("\nDone!")
