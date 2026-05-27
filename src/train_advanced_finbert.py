"""
Advanced FinBERT Training Pipeline
===================================
Integrates 3 advanced techniques for financial sentiment classification:
1. Back-Translation Augmentation (data-level)
2. Extended Pre-training with MLM + Topic multi-task (model-level)
3. Fine-tuning with Multi-task Learning + GPU augmentation (training-level)

Designed to run on Kaggle with GPU. Single self-contained script.
Usage: python train_advanced_finbert.py
"""

import os
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm

# ---------------------------------------------------------------------------
# CONFIGURATION - Easy hyperparameter tuning for experiments
# ---------------------------------------------------------------------------
CONFIG = {
    # Paths
    "train_path": "/kaggle/input/datasets/franciscomiguel/text-mining-preprocessed/train_normalized.csv",
    "test_path": "/kaggle/input/datasets/franciscomiguel/text-mining-preprocessed/test_normalized.csv",
    "topics_path": "/kaggle/working/topics.npy",
    "augmented_path": "preprocessed_data/train_augmented.csv",
    "model_name": "ProsusAI/finbert",

    # Phase 1: Back-Translation
    "aug_pivot_langs": ["de", "fr"],          # Pivot languages for back-translation
    "aug_batch_size": 10,                     # Batch size for translation API calls
    "aug_sleep_min": 0.5,                     # Min sleep between batches (rate limiting)
    "aug_sleep_max": 1.0,                     # Max sleep between batches
    "aug_max_retries": 3,                     # Max retries with exponential backoff
    "aug_target_bearish": 2900,               # Target count for Bearish class
    "aug_target_bullish": 3400,               # Target count for Bullish class

    # Phase 2: Extended Pre-training (MLM + Topic)
    "pretrain_epochs": 3,
    "pretrain_lr": 1e-5,                      # Very low LR to avoid catastrophic forgetting
    "pretrain_batch_size": 32,
    "pretrain_mlm_prob": 0.30,                # 30% masking - aggressive but effective for domain
    "pretrain_mlm_weight": 0.7,               # MLM loss contribution
    "pretrain_topic_weight": 0.3,             # Topic loss contribution
    "pretrain_max_length": 128,
    "freeze_except_last_n": 2,                # Unfreeze only last N encoder layers

    # Phase 3: Fine-tuning
    "finetune_epochs": 7,
    "finetune_lr": 2e-5,
    "finetune_batch_size": 16,
    "finetune_max_length": 128,
    "finetune_warmup_ratio": 0.10,            # 10% of steps for linear warmup
    "finetune_weight_decay": 0.01,
    "finetune_sentiment_weight": 0.8,         # Sentiment loss contribution
    "finetune_topic_weight": 0.2,             # Topic loss contribution
    "finetune_mask_prob": 0.15,               # GPU augmentation: token masking
    "finetune_delete_prob": 0.10,             # GPU augmentation: token deletion

    # General
    "seed": 42,
    "val_split": 0.20,
    "num_labels": 3,                          # Bearish=0, Bullish=1, Neutral=2

    # Output paths
    "output_model_dir": "outputs/models/advanced_finbert",
    "output_figures_dir": "outputs/figures",
    "output_metrics_dir": "outputs/metrics",
    "output_preds_path": "outputs/pred_advanced.csv",

    # Baseline comparison
    "baseline_acc": 0.8685,
    "baseline_macro_f1": 0.83,
}


def set_seed(seed):
    """Reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Detect GPU and warn if not available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[WARNING] No GPU detected! Training will be very slow on CPU.")
        print("[WARNING] This script is designed for Kaggle GPU environment.")
    return device


def generate_topics(texts):
    """
    Generate topic labels using BERTopic.
    Runs only if topics.npy doesn't already exist at the output path.
    WHY: Topic labels provide an auxiliary training signal that encourages
    richer intermediate representations in the encoder.
    """
    topics_path = CONFIG["topics_path"]
    if os.path.exists(topics_path):
        print(f"[INFO] Topics already exist at {topics_path}. Skipping generation.")
        return

    print("\n" + "=" * 60)
    print("  TOPIC GENERATION (BERTopic)")
    print("=" * 60)

    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
    except ImportError:
        print("[INFO] Installing bertopic...")
        import subprocess
        subprocess.check_call(["pip", "install", "bertopic", "-q"])
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN

    # Configure for speed on Kaggle
    umap_model = UMAP(
        n_neighbors=30, n_components=5, min_dist=0.1,
        metric="cosine", random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=30, min_samples=5,
        metric="euclidean", cluster_selection_method="eom"
    )

    topic_model = BERTopic(
        language="english",
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=20,
        verbose=True,
    )

    print(f"[INFO] Fitting BERTopic on {len(texts)} documents...")
    topics, _ = topic_model.fit_transform(texts)
    topics = np.array(topics)

    # Save to working directory
    os.makedirs(os.path.dirname(topics_path) if os.path.dirname(topics_path) else ".", exist_ok=True)
    np.save(topics_path, topics)
    print(f"[INFO] Topics saved to {topics_path}")
    print(f"[INFO] Found {len(set(topics))} unique topics (including outliers=-1)")
    print(f"[INFO] Topic distribution: {dict(zip(*np.unique(topics, return_counts=True)))}")


def sanity_check_topics():
    """
    Run at startup to validate topic labels won't cause CUDA asserts.
    BERTopic assigns -1 to outliers, which crashes cross_entropy.
    Fix: remap -1 → a valid 'outlier' class.
    """
    print("\n[SANITY CHECK] Validating topic labels...")
    topics_path = CONFIG["topics_path"]
    if not os.path.exists(topics_path):
        print("  → topics.npy not found. Topic task will be skipped. OK.")
        return

    topics = np.load(topics_path)
    print(f"  → Loaded {len(topics)} topic labels")
    print(f"  → Unique values: min={topics.min()}, max={topics.max()}, unique={len(np.unique(topics))}")

    if topics.min() < 0:
        print(f"  → [FIX] Found negative topic IDs (BERTopic outliers = -1).")
        print(f"  → Remapping: -1 → {topics.max() + 1} (new 'outlier' class)")
        topics[topics < 0] = topics.max() + 1
        np.save(topics_path, topics)
        print(f"  → Saved fixed topics. New range: [{topics.min()}, {topics.max()}]")
    else:
        print("  → All topic IDs >= 0. OK.")

    # Verify a small cross_entropy test
    num_topics = int(topics.max()) + 1
    test_logits = torch.randn(4, num_topics)
    test_labels = torch.tensor(topics[:4], dtype=torch.long)
    try:
        loss = F.cross_entropy(test_logits, test_labels)
        print(f"  → Cross-entropy test passed (loss={loss.item():.4f}). OK.")
    except Exception as e:
        print(f"  → [ERROR] Cross-entropy test FAILED: {e}")
        raise RuntimeError("Topic labels are invalid. Cannot proceed.")


# ===========================================================================
# PHASE 1: BACK-TRANSLATION AUGMENTATION
# ===========================================================================
# WHY: Financial sentiment datasets are heavily imbalanced (Neutral dominates).
# Back-translation creates semantically equivalent paraphrases that preserve
# the original sentiment while introducing lexical diversity. This is superior
# to simple synonym replacement because it restructures sentences naturally.
# ===========================================================================

def back_translate_batch(texts, pivot_lang, translator_forward, translator_back):
    """
    Translate a batch of texts: EN → pivot_lang → EN.
    Uses translate_batch() for efficiency — single API call per batch.
    This creates natural paraphrases that preserve meaning but vary wording.
    """
    try:
        # Forward: English → Pivot language (batch)
        translated = translator_forward.translate_batch(texts)
        # Backward: Pivot language → English (batch)
        back_translated = translator_back.translate_batch(translated)
        return back_translated
    except Exception as e:
        print(f"  [WARN] Translation error: {e}")
        return None


def simple_augment_fallback(text):
    """
    Fallback augmentation if deep-translator is not available.
    Uses random word shuffling within sentence structure to create variety.
    Not as good as back-translation but better than nothing.
    """
    words = text.split()
    if len(words) < 4:
        return text
    # Randomly swap 20% of adjacent word pairs
    augmented = words.copy()
    n_swaps = max(1, len(augmented) // 5)
    for _ in range(n_swaps):
        idx = random.randint(0, len(augmented) - 2)
        augmented[idx], augmented[idx + 1] = augmented[idx + 1], augmented[idx]
    return " ".join(augmented)


def run_phase1_augmentation():
    """
    Phase 1: Augment minority classes via back-translation.
    Only runs if augmented file doesn't already exist (caching).
    """
    print("\n" + "=" * 60)
    print("  PHASE 1: BACK-TRANSLATION AUGMENTATION")
    print("=" * 60)

    aug_path = CONFIG["augmented_path"]
    os.makedirs(os.path.dirname(aug_path), exist_ok=True)

    # Skip if already computed (expensive API calls)
    if os.path.exists(aug_path):
        print(f"[INFO] Augmented data already exists at {aug_path}. Skipping Phase 1.")
        return pd.read_csv(aug_path)

    # Load training data
    train_df = pd.read_csv(CONFIG["train_path"])
    print(f"[INFO] Original training data: {len(train_df)} samples")
    print(f"[INFO] Class distribution:\n{train_df['label'].value_counts().to_string()}")

    # Identify minority classes to augment
    # Bearish (0): ~1,442 → target ~2,900 (need ~1,458 more)
    # Bullish (1): ~1,923 → target ~3,400 (need ~1,477 more)
    label_counts = train_df["label"].value_counts().to_dict()
    bearish_count = label_counts.get(0, 0)
    bullish_count = label_counts.get(1, 0)

    bearish_needed = max(0, CONFIG["aug_target_bearish"] - bearish_count)
    bullish_needed = max(0, CONFIG["aug_target_bullish"] - bullish_count)

    print(f"\n[INFO] Augmentation targets:")
    print(f"  Bearish (0): {bearish_count} → {CONFIG['aug_target_bearish']} (need {bearish_needed} more)")
    print(f"  Bullish (1): {bullish_count} → {CONFIG['aug_target_bullish']} (need {bullish_needed} more)")

    # Try to import deep-translator
    use_deep_translator = True
    try:
        from deep_translator import GoogleTranslator
        print("[INFO] deep-translator available. Using back-translation.")
    except ImportError:
        use_deep_translator = False
        print("[WARNING] deep-translator not installed!")
        print("[WARNING] Install with: pip install deep-translator")
        print("[WARNING] Falling back to simple word-swap augmentation.")

    augmented_rows = []

    for label, needed in [(0, bearish_needed), (1, bullish_needed)]:
        if needed <= 0:
            continue

        label_name = "Bearish" if label == 0 else "Bullish"
        subset = train_df[train_df["label"] == label]
        texts = subset["text"].tolist()

        print(f"\n[INFO] Augmenting {label_name} class ({needed} samples needed)...")

        if use_deep_translator:
            # Back-translate through each pivot language
            # Each pivot produces one augmented version per sample
            for pivot in CONFIG["aug_pivot_langs"]:
                translator_fwd = GoogleTranslator(source="en", target=pivot)
                translator_bwd = GoogleTranslator(source=pivot, target="en")

                # Sample texts to augment (may need fewer than full set per language)
                samples_per_lang = needed // len(CONFIG["aug_pivot_langs"])
                sample_indices = np.random.choice(
                    len(texts), size=min(samples_per_lang, len(texts)), replace=True
                )
                sampled_texts = [texts[i] for i in sample_indices]

                batch_size = CONFIG["aug_batch_size"]
                augmented_texts = []

                for i in tqdm(range(0, len(sampled_texts), batch_size),
                              desc=f"  {label_name} via {pivot}"):
                    batch = sampled_texts[i:i + batch_size]

                    # Retry with exponential backoff
                    for attempt in range(CONFIG["aug_max_retries"]):
                        result = back_translate_batch(batch, pivot, translator_fwd, translator_bwd)
                        if result is not None:
                            augmented_texts.extend(result)
                            break
                        wait = (2 ** attempt) + random.random()
                        print(f"    Retry {attempt+1}, waiting {wait:.1f}s...")
                        time.sleep(wait)
                    else:
                        # All retries failed, use fallback for this batch
                        augmented_texts.extend([simple_augment_fallback(t) for t in batch])

                    # Rate limiting to avoid API throttling
                    sleep_time = random.uniform(CONFIG["aug_sleep_min"], CONFIG["aug_sleep_max"])
                    time.sleep(sleep_time)

                for text in augmented_texts:
                    augmented_rows.append({"text": text, "label": label})

        else:
            # Fallback: simple augmentation (two versions per sample)
            sample_indices = np.random.choice(len(texts), size=needed, replace=True)
            for idx in tqdm(sample_indices, desc=f"  {label_name} (fallback)"):
                aug_text = simple_augment_fallback(texts[idx])
                augmented_rows.append({"text": aug_text, "label": label})

    # Combine original + augmented
    augmented_df = pd.DataFrame(augmented_rows)
    combined_df = pd.concat([train_df, augmented_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)

    # Save
    combined_df.to_csv(aug_path, index=False)
    print(f"\n[INFO] Augmented dataset saved: {len(combined_df)} samples")
    print(f"[INFO] New class distribution:\n{combined_df['label'].value_counts().to_string()}")

    return combined_df


# ===========================================================================
# PHASE 2: EXTENDED PRE-TRAINING (MLM + TOPIC CLASSIFICATION)
# ===========================================================================
# WHY: FinBERT was pre-trained on general financial text, but our domain may
# have specific vocabulary/patterns. Extended pre-training (also called
# "domain-adaptive pre-training") helps the model adapt its representations
# to our specific corpus. Adding a topic classification auxiliary task provides
# additional structural signal about document categories, which helps the model
# learn more discriminative features. Freezing lower layers prevents catastrophic
# forgetting of general language understanding while allowing top layers to
# specialize.
# ===========================================================================

class MLMTopicDataset(Dataset):
    """Dataset for multi-task pre-training: MLM + Topic classification."""

    def __init__(self, texts, tokenizer, topics=None, max_length=128, mlm_prob=0.3):
        self.texts = texts
        self.tokenizer = tokenizer
        self.topics = topics
        self.max_length = max_length
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Create MLM labels: mask random tokens
        labels = input_ids.clone()
        # Probability matrix for masking (don't mask special tokens)
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask[input_ids == self.tokenizer.pad_token_id] = True
        special_tokens_mask[input_ids == self.tokenizer.cls_token_id] = True
        special_tokens_mask[input_ids == self.tokenizer.sep_token_id] = True

        prob_matrix = torch.full(input_ids.shape, self.mlm_prob)
        prob_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(prob_matrix).bool()

        # Only compute loss on masked tokens
        labels[~masked_indices] = -100

        # 80% of masked → [MASK], 10% → random, 10% → unchanged
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mlm_labels": labels,
        }

        if self.topics is not None:
            item["topic_label"] = torch.tensor(self.topics[idx], dtype=torch.long)

        return item


class TopicHead(nn.Module):
    """
    Topic classification head.
    Architecture: Linear → GELU → Dropout → Linear
    WHY GELU: Smoother than ReLU, works better with transformer representations.
    """

    def __init__(self, hidden_size=768, intermediate_size=384, num_topics=10, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(intermediate_size, num_topics)

    def forward(self, pooled_output):
        x = self.dense(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def freeze_model_layers(model, unfreeze_last_n=2):
    """
    Freeze all layers except the last N encoder layers and the pooler.
    WHY: Lower layers capture general language features (syntax, morphology)
    that transfer well. Upper layers capture task-specific semantics that
    need adaptation. Freezing lower layers also reduces memory and speeds training.
    """
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last N encoder layers
    num_layers = len(model.encoder.layer)
    for i in range(num_layers - unfreeze_last_n, num_layers):
        for param in model.encoder.layer[i].parameters():
            param.requires_grad = True

    # Unfreeze pooler (needed for [CLS] representation)
    for param in model.pooler.parameters():
        param.requires_grad = True

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


def unfreeze_all(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def run_phase2_pretraining(train_texts, device):
    """
    Phase 2: Extended pre-training with MLM + optional Topic classification.
    Adapts FinBERT representations to our specific financial corpus.
    """
    print("\n" + "=" * 60)
    print("  PHASE 2: EXTENDED PRE-TRAINING (MLM + TOPIC)")
    print("=" * 60)

    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained(CONFIG["model_name"])
    bert_model = BertModel.from_pretrained(CONFIG["model_name"]).to(device)

    # Freeze lower layers to prevent catastrophic forgetting
    print(f"[INFO] Freezing all except last {CONFIG['freeze_except_last_n']} encoder layers...")
    freeze_model_layers(bert_model, unfreeze_last_n=CONFIG["freeze_except_last_n"])

    # Load topics if available
    topics = None
    num_topics = 0
    topic_head = None
    if os.path.exists(CONFIG["topics_path"]):
        topics = np.load(CONFIG["topics_path"])
        # Topics were assigned to original corpus only; augmented texts don't have topic labels.
        # Pad with -100 (ignore index) for augmented samples, then filter during loss.
        # Actually: remap to valid classes and assign outlier class to augmented.
        num_topics = int(topics.max()) + 1
        print(f"[INFO] Loaded {len(topics)} topic labels with {num_topics} unique topics.")
        # Pad augmented samples with the most common topic (or 0) as weak label
        if len(topics) < len(train_texts):
            most_common_topic = int(np.bincount(topics.astype(int)).argmax())
            print(f"[INFO] Augmented texts ({len(train_texts)}) > topics ({len(topics)}). "
                  f"Padding with topic={most_common_topic}.")
            topics = np.pad(topics, (0, len(train_texts) - len(topics)),
                           constant_values=most_common_topic)
        elif len(topics) > len(train_texts):
            topics = topics[:len(train_texts)]
        # Final validation: ensure all values are in [0, num_topics)
        topics = np.clip(topics, 0, num_topics - 1)
        topic_head = TopicHead(hidden_size=768, num_topics=num_topics).to(device)
    else:
        print("[INFO] topics/topics.npy not found. Skipping topic task (MLM-only).")

    # MLM head (reuse BERT's built-in LM head weights)
    from transformers import BertForMaskedLM
    mlm_model = BertForMaskedLM.from_pretrained(CONFIG["model_name"]).to(device)
    # We'll use our bert_model for encoding and mlm_model's cls head for predictions
    mlm_head = mlm_model.cls.to(device)
    # Sync weights: copy our bert_model's state into mlm prediction head context
    del mlm_model.bert  # Free memory

    # Dataset and DataLoader
    dataset = MLMTopicDataset(
        texts=train_texts,
        tokenizer=tokenizer,
        topics=topics,
        max_length=CONFIG["pretrain_max_length"],
        mlm_prob=CONFIG["pretrain_mlm_prob"],
    )
    dataloader = DataLoader(
        dataset, batch_size=CONFIG["pretrain_batch_size"], shuffle=True, num_workers=2
    )

    # Optimizer - only trainable parameters
    params = list(bert_model.parameters()) + list(mlm_head.parameters())
    if topic_head is not None:
        params += list(topic_head.parameters())
    optimizer = torch.optim.AdamW(
        [p for p in params if p.requires_grad],
        lr=CONFIG["pretrain_lr"],
        weight_decay=0.01,
    )

    # Training loop
    bert_model.train()
    mlm_head.train()
    if topic_head:
        topic_head.train()

    for epoch in range(CONFIG["pretrain_epochs"]):
        total_loss = 0
        mlm_total = 0
        topic_total = 0

        pbar = tqdm(dataloader, desc=f"  Pre-train Epoch {epoch+1}/{CONFIG['pretrain_epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            mlm_labels = batch["mlm_labels"].to(device)

            # Forward through BERT
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state  # (B, seq_len, 768)
            pooled_output = outputs.pooler_output         # (B, 768)

            # MLM loss
            prediction_scores = mlm_head(sequence_output)
            mlm_loss = F.cross_entropy(
                prediction_scores.view(-1, tokenizer.vocab_size),
                mlm_labels.view(-1),
                ignore_index=-100,
            )

            loss = CONFIG["pretrain_mlm_weight"] * mlm_loss
            mlm_total += mlm_loss.item()

            # Topic loss (if available)
            if topic_head is not None and "topic_label" in batch:
                topic_labels = batch["topic_label"].to(device)
                topic_logits = topic_head(pooled_output)
                topic_loss = F.cross_entropy(topic_logits, topic_labels)
                loss += CONFIG["pretrain_topic_weight"] * topic_loss
                topic_total += topic_loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in params if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        n_batches = len(dataloader)
        msg = f"  Epoch {epoch+1} - Total Loss: {total_loss/n_batches:.4f}, MLM: {mlm_total/n_batches:.4f}"
        if topic_head:
            msg += f", Topic: {topic_total/n_batches:.4f}"
        print(msg)

    # Unfreeze all layers for fine-tuning phase
    print("\n[INFO] Pre-training complete. Unfreezing all layers for fine-tuning...")
    unfreeze_all(bert_model)

    return bert_model, tokenizer, topic_head, num_topics


# ===========================================================================
# PHASE 3: FINE-TUNING WITH MULTI-TASK LEARNING
# ===========================================================================
# WHY: Multi-task learning provides regularization through auxiliary objectives.
# The topic prediction task forces the model to maintain awareness of document
# structure, which helps prevent overfitting to spurious sentiment correlations.
# GPU augmentation (masking/deletion) during training acts as a regularizer,
# making the model robust to noisy/incomplete inputs common in financial text.
# ===========================================================================

class SentimentTopicDataset(Dataset):
    """Dataset for multi-task fine-tuning: Sentiment + Topic."""

    def __init__(self, texts, labels, tokenizer, topics=None, max_length=128,
                 augment=False, mask_prob=0.15, delete_prob=0.10):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.topics = topics
        self.max_length = max_length
        self.augment = augment
        self.mask_prob = mask_prob
        self.delete_prob = delete_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # GPU augmentation: applied only during training
        # WHY: Acts as dropout at the input level, forcing the model to rely on
        # multiple cues rather than memorizing specific token patterns
        if self.augment:
            input_ids, attention_mask = self._apply_augmentation(input_ids, attention_mask)

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.topics is not None:
            item["topic_label"] = torch.tensor(self.topics[idx], dtype=torch.long)
        return item

    def _apply_augmentation(self, input_ids, attention_mask):
        """
        On-the-fly augmentation on GPU-ready tensors.
        - 15% token masking: replaces tokens with [MASK]
        - 10% token deletion: removes tokens and shifts sequence
        """
        special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_mask[input_ids == self.tokenizer.pad_token_id] = True
        special_mask[input_ids == self.tokenizer.cls_token_id] = True
        special_mask[input_ids == self.tokenizer.sep_token_id] = True

        # Token masking
        mask_candidates = ~special_mask & (attention_mask == 1)
        mask_probs = torch.full_like(input_ids, self.mask_prob, dtype=torch.float)
        mask_probs[~mask_candidates] = 0.0
        mask_indices = torch.bernoulli(mask_probs).bool()
        input_ids[mask_indices] = self.tokenizer.mask_token_id

        # Token deletion (shift remaining tokens left, pad at end)
        delete_probs = torch.full_like(input_ids, self.delete_prob, dtype=torch.float)
        delete_probs[special_mask] = 0.0
        delete_probs[mask_indices] = 0.0  # Don't delete already-masked tokens
        delete_indices = torch.bernoulli(delete_probs).bool()

        if delete_indices.any():
            keep_mask = ~delete_indices
            kept_ids = input_ids[keep_mask]
            kept_attn = attention_mask[keep_mask]
            # Pad back to max_length
            pad_len = len(input_ids) - len(kept_ids)
            input_ids = torch.cat([kept_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            attention_mask = torch.cat([kept_attn, torch.zeros(pad_len, dtype=torch.long)])

        return input_ids, attention_mask


class MultiTaskFinBERT(nn.Module):
    """
    Multi-task model with two heads:
    - Sentiment classification (primary task)
    - Topic classification (auxiliary task for regularization)
    """

    def __init__(self, bert_model, num_labels=3, topic_head=None):
        super().__init__()
        self.bert = bert_model
        # Sentiment head: simple linear for fine-grained control
        self.sentiment_head = nn.Linear(768, num_labels)
        # Topic head: reused from Phase 2 (transfer learned topic representations)
        self.topic_head = topic_head

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS] representation

        sentiment_logits = self.sentiment_head(pooled)

        topic_logits = None
        if self.topic_head is not None:
            topic_logits = self.topic_head(pooled)

        return sentiment_logits, topic_logits


def compute_class_weights(labels):
    """
    Compute inverse frequency weights for weighted cross-entropy.
    WHY: Neutral class dominates; without weighting, the model achieves high
    accuracy by simply predicting Neutral for everything. Inverse frequency
    forces equal attention to minority classes.
    """
    counts = Counter(labels)
    total = sum(counts.values())
    weights = torch.zeros(CONFIG["num_labels"])
    for label, count in counts.items():
        weights[label] = total / (CONFIG["num_labels"] * count)
    return weights


def get_linear_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps):
    """
    Linear warmup then linear decay.
    WHY: Warmup prevents early divergence when gradients are noisy at start.
    Linear decay helps convergence at the end of training.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) /
                   float(max(1, num_training_steps - num_warmup_steps)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_phase3_finetuning(bert_model, tokenizer, topic_head, num_topics, train_df, device):
    """
    Phase 3: Fine-tune with multi-task learning (sentiment + topic) and GPU augmentation.
    """
    print("\n" + "=" * 60)
    print("  PHASE 3: FINE-TUNING (MULTI-TASK + GPU AUGMENTATION)")
    print("=" * 60)

    # Stratified split: augmented for training, original distribution for validation
    # WHY: Validate on original distribution to get unbiased performance estimate
    texts = train_df["text"].tolist()
    labels = train_df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels,
        test_size=CONFIG["val_split"],
        stratify=labels,
        random_state=CONFIG["seed"],
    )

    print(f"[INFO] Train: {len(train_texts)}, Validation: {len(val_texts)}")
    print(f"[INFO] Train distribution: {Counter(train_labels)}")
    print(f"[INFO] Val distribution: {Counter(val_labels)}")

    # Load topics for training samples (if available)
    train_topics = None
    val_topics = None
    if num_topics > 0 and os.path.exists(CONFIG["topics_path"]):
        all_topics = np.load(CONFIG["topics_path"])
        if len(all_topics) >= len(texts):
            all_topics_subset = all_topics[:len(texts)]
            # Split same way (use indices)
            indices = list(range(len(texts)))
            train_idx, val_idx = train_test_split(
                indices, test_size=CONFIG["val_split"],
                stratify=labels, random_state=CONFIG["seed"]
            )
            train_topics = all_topics_subset[train_idx]
            val_topics = all_topics_subset[val_idx]

    # Datasets
    train_dataset = SentimentTopicDataset(
        train_texts, train_labels, tokenizer, topics=train_topics,
        max_length=CONFIG["finetune_max_length"],
        augment=True,  # GPU augmentation ON for training
        mask_prob=CONFIG["finetune_mask_prob"],
        delete_prob=CONFIG["finetune_delete_prob"],
    )
    val_dataset = SentimentTopicDataset(
        val_texts, val_labels, tokenizer, topics=val_topics,
        max_length=CONFIG["finetune_max_length"],
        augment=False,  # No augmentation for validation
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["finetune_batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["finetune_batch_size"], shuffle=False, num_workers=2)

    # Model
    model = MultiTaskFinBERT(bert_model, num_labels=CONFIG["num_labels"], topic_head=topic_head).to(device)

    # Weighted cross-entropy for sentiment
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"[INFO] Class weights: {class_weights.tolist()}")
    sentiment_criterion = nn.CrossEntropyLoss(weight=class_weights)
    topic_criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["finetune_lr"],
        weight_decay=CONFIG["finetune_weight_decay"],
    )

    # Scheduler with linear warmup
    total_steps = len(train_loader) * CONFIG["finetune_epochs"]
    warmup_steps = int(total_steps * CONFIG["finetune_warmup_ratio"])
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)
    print(f"[INFO] Total steps: {total_steps}, Warmup steps: {warmup_steps}")

    # Training loop
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(CONFIG["finetune_epochs"]):
        model.train()
        total_loss = 0
        train_preds, train_true = [], []

        pbar = tqdm(train_loader, desc=f"  Fine-tune Epoch {epoch+1}/{CONFIG['finetune_epochs']}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            sentiment_logits, topic_logits = model(input_ids, attention_mask)

            # Combined loss
            loss = CONFIG["finetune_sentiment_weight"] * sentiment_criterion(sentiment_logits, labels)

            if topic_logits is not None and "topic_label" in batch:
                topic_labels = batch["topic_label"].to(device)
                loss += CONFIG["finetune_topic_weight"] * topic_criterion(topic_logits, topic_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = sentiment_logits.argmax(dim=-1).cpu().tolist()
            train_preds.extend(preds)
            train_true.extend(batch["label"].tolist())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_preds, val_true = [], []
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_batch = batch["label"].to(device)

                sentiment_logits, _ = model(input_ids, attention_mask)
                loss = sentiment_criterion(sentiment_logits, labels_batch)
                val_loss += loss.item()

                preds = sentiment_logits.argmax(dim=-1).cpu().tolist()
                val_preds.extend(preds)
                val_true.extend(batch["label"].tolist())

        train_acc = accuracy_score(train_true, train_preds)
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")

        print(f"  Epoch {epoch+1} - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val Macro-F1: {val_f1:.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  [NEW BEST] Macro-F1: {val_f1:.4f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)

    print(f"\n[INFO] Best validation Macro-F1: {best_val_f1:.4f}")

    return model, tokenizer, val_texts, val_labels, val_preds


# ===========================================================================
# PHASE 4: EVALUATION
# ===========================================================================

def run_phase4_evaluation(model, tokenizer, val_labels, val_preds, device):
    """
    Phase 4: Comprehensive evaluation with metrics, confusion matrix, and test predictions.
    """
    print("\n" + "=" * 60)
    print("  PHASE 4: EVALUATION")
    print("=" * 60)

    # Create output directories
    os.makedirs(CONFIG["output_figures_dir"], exist_ok=True)
    os.makedirs(CONFIG["output_metrics_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["output_preds_path"]), exist_ok=True)
    os.makedirs(CONFIG["output_model_dir"], exist_ok=True)

    # Classification report
    label_names = ["Bearish", "Bullish", "Neutral"]
    print("\n--- Classification Report ---")
    report = classification_report(val_labels, val_preds, target_names=label_names, digits=4)
    print(report)

    # Metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    report_dict = classification_report(val_labels, val_preds, target_names=label_names, output_dict=True)

    metrics = {
        "accuracy": val_acc,
        "macro_f1": val_f1,
        "per_class": {name: report_dict[name] for name in label_names},
        "config": {k: v for k, v in CONFIG.items() if not k.startswith("output") and not k.startswith("baseline")},
    }

    metrics_path = os.path.join(CONFIG["output_metrics_dir"], "advanced_finbert_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved to {metrics_path}")

    # Confusion matrix
    cm = confusion_matrix(val_labels, val_preds)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Advanced FinBERT - Confusion Matrix")
        cm_path = os.path.join(CONFIG["output_figures_dir"], "cm_advanced_finbert.png")
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {cm_path}")
    except ImportError:
        print("[WARN] matplotlib/seaborn not available. Skipping confusion matrix plot.")
        print(f"  Confusion Matrix:\n{cm}")

    # Test predictions
    if os.path.exists(CONFIG["test_path"]):
        print("\n[INFO] Generating test predictions...")
        test_df = pd.read_csv(CONFIG["test_path"])
        test_texts = test_df["text"].tolist()

        model.eval()
        all_preds = []

        test_dataset = SentimentTopicDataset(
            test_texts, [0] * len(test_texts), tokenizer,
            max_length=CONFIG["finetune_max_length"], augment=False,
        )
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["finetune_batch_size"], shuffle=False)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="  Test predictions"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                sentiment_logits, _ = model(input_ids, attention_mask)
                preds = sentiment_logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)

        # Save predictions
        pred_df = pd.DataFrame({
            "id": range(len(all_preds)),
            "label": all_preds,
        })
        if "id" in test_df.columns:
            pred_df["id"] = test_df["id"].values

        pred_df.to_csv(CONFIG["output_preds_path"], index=False)
        print(f"[INFO] Test predictions saved to {CONFIG['output_preds_path']}")
    else:
        print(f"[WARN] Test file not found at {CONFIG['test_path']}. Skipping test predictions.")

    # Comparison vs baseline
    print("\n" + "-" * 50)
    print("  COMPARISON vs BASELINE")
    print("-" * 50)
    print(f"  {'Metric':<15} {'Baseline':<12} {'Advanced':<12} {'Delta':<10}")
    print(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*10}")

    acc_delta = val_acc - CONFIG["baseline_acc"]
    f1_delta = val_f1 - CONFIG["baseline_macro_f1"]

    print(f"  {'Accuracy':<15} {CONFIG['baseline_acc']:<12.4f} {val_acc:<12.4f} {acc_delta:+.4f}")
    print(f"  {'Macro-F1':<15} {CONFIG['baseline_macro_f1']:<12.4f} {val_f1:<12.4f} {f1_delta:+.4f}")

    if acc_delta > 0:
        print(f"\n  [SUCCESS] Improved accuracy by {acc_delta*100:.2f} percentage points!")
    else:
        print(f"\n  [NOTE] Accuracy delta: {acc_delta*100:.2f}pp. Check per-class improvements.")

    # Save model
    print(f"\n[INFO] Saving model to {CONFIG['output_model_dir']}...")
    torch.save(model.state_dict(), os.path.join(CONFIG["output_model_dir"], "model.pt"))
    tokenizer.save_pretrained(CONFIG["output_model_dir"])
    print("[INFO] Model saved successfully.")

    return metrics


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 60)
    print("  ADVANCED FINBERT TRAINING PIPELINE")
    print("  Extended Pre-training + Multi-task Learning")
    print("=" * 60)

    set_seed(CONFIG["seed"])
    device = get_device()

    # Load training data directly (no back-translation)
    train_df = pd.read_csv(CONFIG["train_path"])
    print(f"[INFO] Training data: {len(train_df)} samples")

    # Generate topic labels using BERTopic (needed for multi-task learning)
    generate_topics(train_df["text"].tolist())

    # Sanity check: fix topic labels BEFORE any training
    sanity_check_topics()

    # Phase 2: Extended Pre-training
    train_texts = train_df["text"].tolist()
    bert_model, tokenizer, topic_head, num_topics = run_phase2_pretraining(train_texts, device)

    # Phase 3: Fine-tuning
    model, tokenizer, val_texts, val_labels, val_preds = run_phase3_finetuning(
        bert_model, tokenizer, topic_head, num_topics, train_df, device
    )

    # Phase 4: Evaluation
    metrics = run_phase4_evaluation(model, tokenizer, val_labels, val_preds, device)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Final Macro-F1: {metrics['macro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
