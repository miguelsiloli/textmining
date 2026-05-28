"""
Ordinal Margin-Based FinBERT for Financial Sentiment Analysis
=============================================================

This script implements a duplicate of the baseline FinBERT approach (simple fine-tuning,
no extended pretraining, no multi-task) but replaces standard cross-entropy with an
ORDINAL MARGIN-BASED LOSS.

WHY ORDINAL MARGIN FOR FINANCIAL SENTIMENT?
--------------------------------------------
Financial sentiment is inherently ordinal: Bearish < Neutral < Bullish.
A standard classifier treats these as unrelated categories — predicting Bearish when
the true label is Bullish is penalized the same as predicting Neutral.

But in reality, confusing Bearish↔Bullish is a MUCH worse error than Bearish↔Neutral.
The ordinal approach:
1. Projects [CLS] embeddings onto a 1D "sentiment axis"
2. Uses learnable thresholds to partition this axis into classes
3. Enforces margins between classes, respecting the natural ordering

This means the model learns a continuous sentiment representation where:
- Bearish samples cluster on the LEFT (low scores)
- Neutral samples cluster in the MIDDLE
- Bullish samples cluster on the RIGHT (high scores)

The margin enforcement ensures clear separation between adjacent classes.

Target: Kaggle GPU environment
Baseline comparison: Standard FinBERT (86.85% acc, 0.83 macro F1)
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION
# ==============================================================================

CONFIG = {
    "model_name": "ProsusAI/finbert",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 9,
    "lr": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "margin": 0.5,
    "min_threshold_gap": 0.5,
    "seed": 42,
    # Loss weights
    "hinge_weight": 1.0,
    "ce_weight": 0.3,
    "threshold_reg_weight": 0.1,
}

# Paths
TRAIN_PATH = "/kaggle/input/datasets/franciscomiguel/text-mining-preprocessed/train_normalized.csv"
TEST_PATH = "/kaggle/input/datasets/franciscomiguel/text-mining-preprocessed/test_normalized.csv"
OUTPUT_DIR = "outputs"
FIGURES_DIR = "outputs/figures"
MODELS_DIR = "outputs/models"

LABEL_MAP = {0: "Bearish", 1: "Neutral", 2: "Bullish"}


# ==============================================================================
# REPRODUCIBILITY
# ==============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================================
# DATASET
# ==============================================================================

class SentimentDataset(Dataset):
    """Simple tokenization dataset — baseline approach, no augmentation."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ==============================================================================
# MODEL: Ordinal Margin FinBERT
# ==============================================================================

class OrdinalMarginFinBERT(nn.Module):
    """
    Instead of 3 independent logits, project [CLS] to a 1D latent sentiment axis
    with learnable thresholds + margin enforcement.

    The sentiment axis naturally orders classes:
        Bearish (low) --- θ₁ --- Neutral (mid) --- θ₂ --- Bullish (high)
    """

    def __init__(self, model_name="ProsusAI/finbert", num_classes=3, margin=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        # Project to 1D sentiment score
        self.score_head = nn.Linear(768, 1)
        # K-1 = 2 learnable thresholds (initialized ordered: θ₁ < θ₂)
        self.thresholds = nn.Parameter(torch.tensor([-1.0, 1.0]))
        self.margin = margin
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        score = self.score_head(pooled).squeeze(-1)  # (batch,)
        return score

    def predict_class(self, score):
        """
        Assign class based on threshold positions:
        - Bearish if score < θ₁
        - Neutral if θ₁ ≤ score < θ₂
        - Bullish if score ≥ θ₂
        """
        preds = torch.zeros_like(score, dtype=torch.long)
        preds[score >= self.thresholds[0]] = 1
        preds[score >= self.thresholds[1]] = 2
        return preds

    def get_class_probabilities(self, score):
        """
        Compute class probabilities using cumulative link model:
        P(y > k) = σ(score - θ_k)
        """
        # P(y > 0) = σ(score - θ₁)
        p_gt_0 = torch.sigmoid(score - self.thresholds[0])
        # P(y > 1) = σ(score - θ₂)
        p_gt_1 = torch.sigmoid(score - self.thresholds[1])

        # Class probabilities
        p_class_0 = 1 - p_gt_0          # P(y = 0) = 1 - P(y > 0)
        p_class_1 = p_gt_0 - p_gt_1     # P(y = 1) = P(y > 0) - P(y > 1)
        p_class_2 = p_gt_1              # P(y = 2) = P(y > 1)

        # Stack and clamp for numerical stability
        probs = torch.stack([p_class_0, p_class_1, p_class_2], dim=-1)
        probs = probs.clamp(min=1e-7)
        return probs


# ==============================================================================
# LOSS FUNCTION: Ordinal Margin Loss
# ==============================================================================

class OrdinalMarginLoss(nn.Module):
    """
    Combined loss for ordinal classification:
    1. Ordinal hinge loss — enforces scores on correct side of thresholds with margin
    2. Threshold ordering regularization — ensures θ₁ < θ₂
    3. Distance-weighted cross-entropy — soft probabilistic loss via cumulative model
    """

    def __init__(self, margin=0.5, min_gap=0.5, hinge_weight=1.0,
                 ce_weight=0.3, reg_weight=0.1):
        super().__init__()
        self.margin = margin
        self.min_gap = min_gap
        self.hinge_weight = hinge_weight
        self.ce_weight = ce_weight
        self.reg_weight = reg_weight

    def forward(self, score, labels, model):
        thresholds = model.thresholds

        # --- 1. Ordinal Hinge Loss ---
        # Enforce: score is on the correct side of each threshold with margin
        hinge_loss = torch.zeros_like(score)

        # Bearish (y=0): score should be BELOW θ₁ by at least margin
        mask_0 = (labels == 0)
        if mask_0.any():
            hinge_loss[mask_0] = F.relu(score[mask_0] - thresholds[0] + self.margin)

        # Neutral (y=1): score should be ABOVE θ₁ and BELOW θ₂ by at least margin
        mask_1 = (labels == 1)
        if mask_1.any():
            hinge_loss[mask_1] = (
                F.relu(thresholds[0] - score[mask_1] + self.margin) +
                F.relu(score[mask_1] - thresholds[1] + self.margin)
            )

        # Bullish (y=2): score should be ABOVE θ₂ by at least margin
        mask_2 = (labels == 2)
        if mask_2.any():
            hinge_loss[mask_2] = F.relu(thresholds[1] - score[mask_2] + self.margin)

        hinge_loss = hinge_loss.mean()

        # --- 2. Threshold Ordering Regularization ---
        # Ensure θ₁ < θ₂ with a minimum gap
        threshold_reg = F.relu(thresholds[0] - thresholds[1] + self.min_gap)

        # --- 3. Distance-Weighted Cross-Entropy ---
        # Use cumulative link probabilities for a soft CE loss
        probs = model.get_class_probabilities(score)
        log_probs = torch.log(probs)
        ce_loss = F.nll_loss(log_probs, labels)

        # --- Combined Loss ---
        total_loss = (
            self.hinge_weight * hinge_loss +
            self.reg_weight * threshold_reg +
            self.ce_weight * ce_loss
        )

        return total_loss, {
            "hinge": hinge_loss.item(),
            "ce": ce_loss.item(),
            "reg": threshold_reg.item(),
            "total": total_loss.item(),
        }


# ==============================================================================
# TRAINING
# ==============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    loss_components = {"hinge": 0, "ce": 0, "reg": 0}

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        score = model(input_ids, attention_mask)
        loss, components = criterion(score, labels, model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += components[k]

        preds = model.predict_class(score.detach())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    n = len(dataloader)
    avg_loss = total_loss / n
    loss_components = {k: v / n for k, v in loss_components.items()}
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, f1, loss_components


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            score = model(input_ids, attention_mask)
            loss, _ = criterion(score, labels, model)

            total_loss += loss.item()
            preds = model.predict_class(score)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(score.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc, f1, np.array(all_preds), np.array(all_labels), np.array(all_scores)


def print_score_distribution(scores, labels, thresholds):
    """Print score distribution per class — shows ordinal structure."""
    print("\n  Score distribution per class:")
    print(f"  {'Class':<10} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*42}")
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            s = scores[mask]
            print(f"  {LABEL_MAP[c]:<10} {s.mean():>8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f}")
    print(f"  Thresholds: θ₁={thresholds[0]:.3f}, θ₂={thresholds[1]:.3f}")


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def plot_score_distribution(scores, labels, thresholds, save_path):
    """
    Plot score histograms per class — should show ordered structure:
    Bearish (left) < Neutral (middle) < Bullish (right)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Bearish": "#e74c3c", "Neutral": "#95a5a6", "Bullish": "#27ae60"}

    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            ax.hist(scores[mask], bins=30, alpha=0.6, label=LABEL_MAP[c],
                    color=colors[LABEL_MAP[c]], density=True)

    # Plot thresholds
    ax.axvline(thresholds[0], color="black", linestyle="--", linewidth=2,
               label=f"θ₁ = {thresholds[0]:.2f}")
    ax.axvline(thresholds[1], color="black", linestyle="-.", linewidth=2,
               label=f"θ₂ = {thresholds[1]:.2f}")

    ax.set_xlabel("Sentiment Score (1D latent axis)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Learned Ordinal Score Distribution per Class\n"
                 "(Bearish ← left | middle → Neutral | right → Bullish)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nScore distribution figure saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(LABEL_MAP.values()),
                yticklabels=list(LABEL_MAP.values()), ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Ordinal Margin FinBERT")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {json.dumps(CONFIG, indent=2)}")

    # --- Create output directories ---
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # --- Load Data ---
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Label distribution:\n{train_df['label'].value_counts().sort_index()}")

    # --- Stratified Split ---
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=train_df["label"].tolist(),
    )
    print(f"\nTrain split: {len(train_texts)}, Val split: {len(val_texts)}")

    # --- Tokenizer & Datasets ---
    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, CONFIG["max_length"])
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, CONFIG["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # --- Model ---
    print("\n" + "=" * 60)
    print("INITIALIZING ORDINAL MARGIN FINBERT")
    print("=" * 60)
    model = OrdinalMarginFinBERT(
        model_name=CONFIG["model_name"],
        margin=CONFIG["margin"],
    ).to(device)

    print(f"Initial thresholds: θ₁={model.thresholds[0].item():.3f}, "
          f"θ₂={model.thresholds[1].item():.3f}")

    # --- Loss & Optimizer ---
    criterion = OrdinalMarginLoss(
        margin=CONFIG["margin"],
        min_gap=CONFIG["min_threshold_gap"],
        hinge_weight=CONFIG["hinge_weight"],
        ce_weight=CONFIG["ce_weight"],
        reg_weight=CONFIG["threshold_reg_weight"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    total_steps = len(train_loader) * CONFIG["epochs"]
    warmup_steps = int(total_steps * CONFIG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # --- Training Loop ---
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    best_val_f1 = 0
    best_epoch = 0
    history = []

    for epoch in range(CONFIG["epochs"]):
        print(f"\n{'─' * 60}")
        print(f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        print(f"{'─' * 60}")

        # Train
        train_loss, train_acc, train_f1, loss_comp = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )

        # Evaluate
        val_loss, val_acc, val_f1, val_preds, val_labels_arr, val_scores = evaluate(
            model, val_loader, criterion, device
        )

        # Get current thresholds
        thresholds = model.thresholds.detach().cpu().numpy()

        # Log
        print(f"\n  Train — Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"  Loss components — Hinge: {loss_comp['hinge']:.4f} | "
              f"CE: {loss_comp['ce']:.4f} | Reg: {loss_comp['reg']:.4f}")
        print(f"  Thresholds: θ₁={thresholds[0]:.4f}, θ₂={thresholds[1]:.4f} "
              f"(gap={thresholds[1] - thresholds[0]:.4f})")

        # Print score distribution per class
        print_score_distribution(val_scores, val_labels_arr, thresholds)

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_ordinal_finbert.pt"))
            print(f"\n  ★ New best model saved! (Val F1: {val_f1:.4f})")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "theta_1": float(thresholds[0]),
            "theta_2": float(thresholds[1]),
        })

    # --- Load Best Model ---
    print("\n" + "=" * 60)
    print(f"LOADING BEST MODEL (Epoch {best_epoch}, Val F1: {best_val_f1:.4f})")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "best_ordinal_finbert.pt")))
    model.eval()

    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON VALIDATION SET")
    print("=" * 60)
    _, final_acc, final_f1, final_preds, final_labels, final_scores = evaluate(
        model, val_loader, criterion, device
    )

    print(f"\nFinal Val Accuracy: {final_acc:.4f}")
    print(f"Final Val Macro F1: {final_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        final_labels, final_preds,
        target_names=list(LABEL_MAP.values()), digits=4
    ))

    # --- Comparison with Baseline ---
    print("\n" + "=" * 60)
    print("COMPARISON WITH STANDARD FINBERT BASELINE")
    print("=" * 60)
    baseline_acc = 0.8685
    baseline_f1 = 0.83
    print(f"  Standard FinBERT (CE loss):  Acc={baseline_acc:.4f} | Macro F1={baseline_f1:.4f}")
    print(f"  Ordinal Margin FinBERT:      Acc={final_acc:.4f} | Macro F1={final_f1:.4f}")
    acc_diff = final_acc - baseline_acc
    f1_diff = final_f1 - baseline_f1
    print(f"  Difference:                  Acc={acc_diff:+.4f} | Macro F1={f1_diff:+.4f}")
    if final_f1 > baseline_f1:
        print("  ✓ Ordinal margin approach IMPROVES over baseline!")
    else:
        print("  ✗ Ordinal margin approach does not beat baseline (may need tuning)")

    # --- Visualizations ---
    # Score distribution (key visualization for ordinal approach)
    thresholds = model.thresholds.detach().cpu().numpy()
    plot_score_distribution(
        final_scores, final_labels, thresholds,
        os.path.join(FIGURES_DIR, "ordinal_score_distribution.png")
    )

    # Confusion matrix
    plot_confusion_matrix(
        final_labels, final_preds,
        os.path.join(FIGURES_DIR, "ordinal_confusion_matrix.png")
    )

    # --- Learned Score Distribution Analysis ---
    print("\n" + "=" * 60)
    print("LEARNED SENTIMENT SCORE DISTRIBUTION")
    print("=" * 60)
    print("\nThe ordinal model projects all texts onto a 1D sentiment axis.")
    print("If the model learned correctly, classes should be ordered:\n")
    print(f"  Bearish (left) ←── θ₁={thresholds[0]:.3f} ──→ "
          f"Neutral (middle) ←── θ₂={thresholds[1]:.3f} ──→ Bullish (right)\n")
    print_score_distribution(final_scores, final_labels, thresholds)

    # Check ordering is maintained
    class_means = []
    for c in range(3):
        mask = final_labels == c
        if mask.sum() > 0:
            class_means.append(final_scores[mask].mean())
    if len(class_means) == 3 and class_means[0] < class_means[1] < class_means[2]:
        print("\n  ✓ Class means are properly ordered (Bearish < Neutral < Bullish)")
        print("    → The model successfully learned an ordinal sentiment representation!")
    else:
        print(f"\n  ⚠ Class means: {[f'{m:.3f}' for m in class_means]}")
        print("    → Ordering may need further tuning")

    # --- Test Predictions ---
    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)
    test_dataset = SentimentDataset(
        test_df["text"].tolist(),
        [0] * len(test_df),  # dummy labels
        tokenizer,
        CONFIG["max_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    test_preds = []
    test_scores_all = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test predictions"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            score = model(input_ids, attention_mask)
            preds = model.predict_class(score)
            test_preds.extend(preds.cpu().numpy())
            test_scores_all.extend(score.cpu().numpy())

    # Save predictions
    test_df_out = test_df.copy()
    test_df_out["label"] = test_preds
    test_df_out["sentiment_score"] = test_scores_all
    test_df_out["label_name"] = test_df_out["label"].map(LABEL_MAP)
    predictions_path = os.path.join(OUTPUT_DIR, "ordinal_finbert_predictions.csv")
    test_df_out.to_csv(predictions_path, index=False)
    print(f"Test predictions saved to: {predictions_path}")
    print(f"Prediction distribution:\n{test_df_out['label_name'].value_counts()}")

    # --- Save Metrics ---
    metrics = {
        "model": "OrdinalMarginFinBERT",
        "config": CONFIG,
        "best_epoch": best_epoch,
        "val_accuracy": float(final_acc),
        "val_macro_f1": float(final_f1),
        "baseline_accuracy": baseline_acc,
        "baseline_macro_f1": baseline_f1,
        "acc_improvement": float(acc_diff),
        "f1_improvement": float(f1_diff),
        "learned_thresholds": {
            "theta_1": float(thresholds[0]),
            "theta_2": float(thresholds[1]),
            "gap": float(thresholds[1] - thresholds[0]),
        },
        "training_history": history,
    }
    metrics_path = os.path.join(OUTPUT_DIR, "ordinal_finbert_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
