"""DistilBERT Quick Training - Ultra-fast with stratified 20% sample."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"CUDA available: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using: {device}")

# Load data & stratified sample 20% for speed
df = pd.read_csv('data/train.csv')
print(f"Full dataset: {len(df)} rows")
print(f"Label distribution:\n{df['label'].value_counts().sort_index()}")

df_sample, _ = train_test_split(df, train_size=0.2, stratify=df['label'], random_state=42)
print(f"\nSampled: {len(df_sample)} rows")

# Train/val split
train_df, val_df = train_test_split(df_sample, test_size=0.2, stratify=df_sample['label'], random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df[['text', 'label']].reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df[['text', 'label']].reset_index(drop=True))

train_ds = train_ds.map(tokenize, batched=True, batch_size=256)
val_ds = val_ds.map(tokenize, batched=True, batch_size=256)

train_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
val_ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
print("Tokenization done.")

# Model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

training_args = TrainingArguments(
    output_dir='models/distilbert_quick',
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=0,
    logging_steps=25,
    report_to='none',
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {'accuracy': accuracy_score(labels, preds)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

print("\n--- Starting Training ---")
trainer.train()

# Evaluation
results = trainer.evaluate()
print(f"\nVal Accuracy: {results['eval_accuracy']:.4f}")

preds = trainer.predict(val_ds)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = val_df['label'].values

label_names = ['Bearish', 'Bullish', 'Neutral']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_names))

# Test predictions
test_df = pd.read_csv('data/test.csv')
test_ds = Dataset.from_pandas(test_df[['text']].reset_index(drop=True))
test_ds = test_ds.map(tokenize, batched=True, batch_size=256)
test_ds.set_format('torch', columns=['input_ids', 'attention_mask'])

test_preds = trainer.predict(test_ds)
test_labels = np.argmax(test_preds.predictions, axis=-1)

os.makedirs('outputs', exist_ok=True)
submission = pd.DataFrame({'label': test_labels})
submission.to_csv('outputs/distilbert_quick_submission.csv', index=False)
print(f"\nTest predictions saved. Distribution:\n{pd.Series(test_labels).value_counts().sort_index()}")
print("\nDone!")
