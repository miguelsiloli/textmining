"""
Extended Pretraining Module for FinBERT on Financial Tweets

This module implements domain-adaptive pretraining with:
1. Masked Language Modeling (MLM) with hard masking
2. Topic Classification as auxiliary task (19 topics from BERTopic)
3. GPU-accelerated hard augmentation

Expected Impact: +2-3% accuracy improvement
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, BertModel
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from typing import Tuple, Optional
import pickle


class HardGPUAugmenter:
    """
    GPU-accelerated hard augmentation for pretraining.
    More aggressive than training augmentation to force robustness.
    """
    
    def __init__(self, mask_token_id, pad_token_id, vocab_size, mask_prob=0.30):
        """
        Args:
            mask_token_id: ID for [MASK] token
            pad_token_id: ID for [PAD] token
            vocab_size: Size of vocabulary
            mask_prob: Probability of masking each token (30% for hard augmentation)
        """
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.mask_prob = mask_prob
    
    def hard_augment(self, input_ids, attention_mask):
        """
        Apply HARD augmentation strategies:
        1. Random masking (30% probability)
        2. Random token replacement (15% probability)
        3. Token deletion (20% probability)
        4. Token swapping (10% probability)
        
        All operations on GPU for speed.
        """
        # Clone to avoid modifying original
        aug_input_ids = input_ids.clone()
        aug_attention_mask = attention_mask.clone()
        
        batch_size, seq_len = aug_input_ids.shape
        
        # Create mask for augmentable tokens (not [CLS], [SEP], [PAD], special tokens)
        special_tokens_mask = (
            (aug_input_ids == 101) |  # [CLS]
            (aug_input_ids == 102) |  # [SEP]
            (aug_input_ids == 0) |    # [PAD]
            (aug_input_ids >= self.vocab_size - 10)  # Special tokens [TICKER], etc.
        )
        augmentable_mask = ~special_tokens_mask & (aug_attention_mask == 1)
        
        # 1. Random Masking (30% probability)
        mask_prob_tensor = torch.rand(batch_size, seq_len, device=aug_input_ids.device)
        should_mask = (mask_prob_tensor < self.mask_prob) & augmentable_mask
        aug_input_ids[should_mask] = self.mask_token_id
        
        # 2. Random Token Replacement (15% probability)
        replace_prob_tensor = torch.rand(batch_size, seq_len, device=aug_input_ids.device)
        should_replace = (replace_prob_tensor < 0.15) & augmentable_mask & ~should_mask
        random_tokens = torch.randint(
            low=1000,  # Start from common vocab
            high=min(20000, self.vocab_size - 10),  # Avoid special tokens
            size=(batch_size, seq_len),
            device=aug_input_ids.device
        )
        aug_input_ids[should_replace] = random_tokens[should_replace]
        
        # 3. Token Deletion (20% probability)
        delete_prob_tensor = torch.rand(batch_size, seq_len, device=aug_input_ids.device)
        should_delete = (delete_prob_tensor < 0.20) & augmentable_mask
        aug_attention_mask[should_delete] = 0
        
        # 4. Token Swapping (10% probability - swap adjacent tokens)
        swap_prob_tensor = torch.rand(batch_size, seq_len - 1, device=aug_input_ids.device)
        should_swap = (swap_prob_tensor < 0.10) & augmentable_mask[:, :-1] & augmentable_mask[:, 1:]
        
        # Perform swaps
        for b in range(batch_size):
            swap_positions = should_swap[b].nonzero(as_tuple=True)[0]
            for pos in swap_positions:
                # Swap tokens at pos and pos+1
                temp = aug_input_ids[b, pos].clone()
                aug_input_ids[b, pos] = aug_input_ids[b, pos + 1]
                aug_input_ids[b, pos + 1] = temp
        
        return aug_input_ids, aug_attention_mask


class PretrainingDataset(Dataset):
    """Dataset for MLM + Topic Classification pretraining"""
    
    def __init__(
        self,
        texts,
        topics,
        tokenizer,
        augmenter,
        max_length=128,
        mlm_probability=0.15
    ):
        """
        Args:
            texts: List of text samples
            topics: List of topic IDs (from BERTopic)
            tokenizer: BERT tokenizer
            augmenter: HardGPUAugmenter instance
            max_length: Max sequence length
            mlm_probability: Base MLM masking probability (15%)
        """
        self.texts = texts
        self.topics = topics
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.max_length = max_length
        self.mlm_probability = mlm_probability
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        topic = self.topics[idx] if self.topics[idx] >= 0 else 0  # Map -1 (outliers) to 0
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        # Store original for MLM labels
        original_input_ids = input_ids.clone()
        
        # Apply hard augmentation (30% masking + replacements + deletions + swaps)
        aug_input_ids, aug_attention_mask = self.augmenter.hard_augment(
            input_ids.unsqueeze(0),
            attention_mask.unsqueeze(0)
        )
        
        return {
            'input_ids': aug_input_ids.squeeze(0),
            'attention_mask': aug_attention_mask.squeeze(0),
            'labels': original_input_ids,  # Original tokens for MLM
            'topic_labels': torch.tensor(topic, dtype=torch.long)
        }


class BertForPretraining(nn.Module):
    """
    BERT model for multi-task pretraining:
    1. Masked Language Modeling (MLM)
    2. Topic Classification
    """
    
    def __init__(self, model_name: str, num_topics: int = 20):
        """
        Args:
            model_name: Pretrained BERT model name
            num_topics: Number of topics (19 + 1 for outliers)
        """
        super().__init__()
        
        # Load BERT for MLM
        self.bert_mlm = BertForMaskedLM.from_pretrained(model_name)
        
        # Topic classification head
        self.bert_base = BertModel.from_pretrained(model_name)
        hidden_size = self.bert_base.config.hidden_size
        
        self.topic_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_topics)
        )
    
    def forward(self, input_ids, attention_mask, labels=None, topic_labels=None):
        """
        Forward pass for multi-task pretraining
        
        Returns:
            Dictionary with MLM loss, topic loss, and combined loss
        """
        # 1. MLM Task
        mlm_outputs = self.bert_mlm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        mlm_loss = mlm_outputs.loss
        
        # 2. Topic Classification Task
        bert_outputs = self.bert_base(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        cls_output = bert_outputs.last_hidden_state[:, 0, :]
        topic_logits = self.topic_classifier(cls_output)
        
        # Topic loss
        topic_loss = None
        if topic_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            topic_loss = loss_fct(topic_logits, topic_labels)
        
        # Combined loss (70% MLM, 30% Topic)
        combined_loss = None
        if mlm_loss is not None and topic_loss is not None:
            combined_loss = 0.7 * mlm_loss + 0.3 * topic_loss
        
        return {
            'loss': combined_loss,
            'mlm_loss': mlm_loss,
            'topic_loss': topic_loss,
            'topic_logits': topic_logits
        }


def load_topic_assignments(topic_model_path: str, texts) -> np.ndarray:
    """
    Load BERTopic model and get topic assignments for texts
    
    Args:
        topic_model_path: Path to saved BERTopic model
        texts: List of texts to assign topics to
        
    Returns:
        Array of topic IDs
    """
    try:
        from bertopic import BERTopic
        
        # Load BERTopic model
        topic_model = BERTopic.load(topic_model_path)
        
        # Get topic assignments (returns topics, probabilities)
        topics, _ = topic_model.transform(texts)
        
        return np.array(topics)
    
    except Exception as e:
        print(f"⚠ Warning: Could not load topic model: {e}")
        print("  Using random topic assignments (pretraining will be less effective)")
        return np.random.randint(0, 19, size=len(texts))


def run_pretraining(
    train_texts,
    model_name: str = 'ProsusAI/finbert',
    topic_model_path: str = './topics/bertopic_model.pkl',
    output_path: str = './pretrained_finbert',
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 5e-5,
    mask_prob: float = 0.30,
    device: str = 'cuda'
):
    """
    Run extended pretraining with MLM + Topic Classification
    
    Args:
        train_texts: List of training texts
        model_name: Base model to start from
        topic_model_path: Path to BERTopic model
        output_path: Where to save pretrained model
        num_epochs: Number of pretraining epochs (2-3 recommended)
        batch_size: Batch size
        learning_rate: Learning rate for pretraining
        mask_prob: Hard masking probability (0.30 = 30%)
        device: Device to use
        
    Returns:
        Pretrained model
    """
    print("="*80)
    print("EXTENDED PRETRAINING: MLM + TOPIC CLASSIFICATION")
    print("="*80)
    
    # Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Load topic assignments
    print("\n[2/7] Loading topic assignments...")
    topics = load_topic_assignments(topic_model_path, train_texts)
    num_topics = len(np.unique(topics[topics >= 0])) + 1  # +1 for outliers
    print(f"  ✓ Loaded {num_topics} topics")
    
    # Initialize augmenter
    print("\n[3/7] Initializing hard augmenter...")
    augmenter = HardGPUAugmenter(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
        mask_prob=mask_prob
    )
    print(f"  ✓ Hard augmentation: {mask_prob*100}% masking + replacements + deletions + swaps")
    
    # Create dataset
    print("\n[4/7] Creating pretraining dataset...")
    dataset = PretrainingDataset(
        texts=train_texts,
        topics=topics,
        tokenizer=tokenizer,
        augmenter=augmenter
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"  ✓ {len(dataset)} samples, {len(dataloader)} batches")
    
    # Initialize model
    print("\n[5/7] Initializing pretraining model...")
    model = BertForPretraining(model_name, num_topics=num_topics)
    model.to(device)
    print(f"  ✓ Model loaded on {device}")
    
    # Setup optimizer
    print("\n[6/7] Setting up optimizer...")
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\n[7/7] Pretraining for {num_epochs} epochs...")
    print("="*80)
    
    model.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-"*80)
        
        total_loss = 0
        total_mlm_loss = 0
        total_topic_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Pretraining")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            topic_labels = batch['topic_labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                topic_labels=topic_labels
            )
            
            loss = outputs['loss']
            mlm_loss = outputs['mlm_loss']
            topic_loss = outputs['topic_loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track losses
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_topic_loss += topic_loss.item()
            
            # Update progress
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mlm': f'{mlm_loss.item():.4f}',
                'topic': f'{topic_loss.item():.4f}'
            })
        
        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        avg_mlm_loss = total_mlm_loss / len(dataloader)
        avg_topic_loss = total_topic_loss / len(dataloader)
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Combined Loss: {avg_loss:.4f}")
        print(f"  MLM Loss: {avg_mlm_loss:.4f}")
        print(f"  Topic Loss: {avg_topic_loss:.4f}")
    
    # Save pretrained model
    print("\n" + "="*80)
    print("Saving pretrained model...")
    model.bert_mlm.bert.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"✓ Pretrained model saved to {output_path}")
    
    print("\n" + "="*80)
    print("✓ PRETRAINING COMPLETE!")
    print("="*80)
    print(f"\n💡 Next Step: Use '{output_path}' as model_name in fine-tuning")
    print("   Expected improvement: +2-3% accuracy")
    print("="*80)
    
    return model


if __name__ == '__main__':
    # Example usage
    import pandas as pd
    
    # Load data
    df = pd.read_csv('../preprocessed_data/train_normalized.csv')
    train_texts = df['text_normalized'].values
    
    # Run pretraining
    model = run_pretraining(
        train_texts=train_texts,
        model_name='ProsusAI/finbert',
        topic_model_path='../topics/bertopic_model.pkl',
        output_path='./pretrained_finbert',
        num_epochs=3,
        batch_size=32,
        learning_rate=5e-5,
        mask_prob=0.30
    )
