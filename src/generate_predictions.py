"""Generate predictions on test set using best available local model (TF-IDF + LogReg)."""

import pandas as pd
import os
import joblib

# Load test data
test = pd.read_csv("data/test.csv")
print(f"Test set: {len(test)} rows")

# Check if predictions.csv exists
pred_path = None
for p in ["predictions.csv", "../predictions.csv"]:
    if os.path.exists(p):
        pred_path = p
        break

if pred_path:
    print(f"Found existing predictions at: {pred_path}")
    preds = pd.read_csv(pred_path)
    print(f"Columns: {list(preds.columns)}")
    # Reformat
    if 'label' in preds.columns:
        labels = preds['label'].values
    elif 'prediction' in preds.columns:
        labels = preds['prediction'].values
    else:
        # Take last column
        labels = preds.iloc[:, -1].values

    # Map string labels to numeric if needed
    label_map = {"Bearish": 0, "Bullish": 1, "Neutral": 2}
    if isinstance(labels[0], str):
        labels = [label_map.get(l, l) for l in labels]

    out = pd.DataFrame({"id": range(len(labels)), "label": labels})
else:
    print("No predictions.csv found. Using TF-IDF + LogReg model.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # Load training data and retrain
    train = pd.read_csv("data/train.csv")

    # Drop rows with missing labels
    train = train.dropna(subset=['label'])
    y_train = train['label'].astype(int).values

    tfidf = TfidfVectorizer(max_features=10000)
    X_train = tfidf.fit_transform(train['text'].fillna(""))
    X_test = tfidf.transform(test['text'].fillna(""))

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    out = pd.DataFrame({"id": range(len(preds)), "label": preds})

os.makedirs("outputs", exist_ok=True)
out.to_csv("outputs/pred_xx.csv", index=False)
print(f"\nSaved: outputs/pred_xx.csv ({len(out)} rows)")
print(f"\nLabel distribution:\n{out['label'].value_counts().sort_index()}")
