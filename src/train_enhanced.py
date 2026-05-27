"""
Enhanced Training Pipeline with Extended Pretraining + Hard Augmentation

This script combines:
1. Extended pretraining (MLM + Topics) - Run FIRST
2. Hard GPU-bound augmentation during fine-tuning
3. Normalized text input

Expected final accuracy: 89-93%
"""

# Imports remain the same as before
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class HarderGPUAugmenter:
    """
    HARDER GPU-accelerated augmentation for fine-tuning.
    More aggressive than baseline to improve robustness.
    """
    
    def __init__(self, mask_token_id, pad_token_id, vocab_size, augment_prob=0.20):
        """
        Args:
            augment_prob: Increased from 0.15 to 0.20 for harder augmentation
        """
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.vocab_size = vocab_size
        self.augment_prob = augment_prob
    
    def augment(self, input_ids, attention_mask):
        """
        Apply HARDER augmentation:
        1. Random masking (20% probability - increased from 15%)
        2. Random deletion (15% probability - increased from 10.5%)
        3. Random replacement (10% probability - NEW!)
        4. Always augment (removed 50% skip chance)
        
        All GPU-bound for speed.
        """
        # Clone to avoid modifying original
        aug_input_ids = input_ids.clone()
        aug_attention_mask = attention_mask.clone()
        
        # Handle both single samples (1D) and batches (2D)
        if aug_input_ids.dim() == 1:
            aug_input_ids = aug_input_ids.unsqueeze(0)
            aug_attention_mask = aug_attention_mask.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, seq_len = aug_input_ids.shape
        
        # Create mask for non-special tokens
        special_tokens_mask = (aug_input_ids == 101) | (aug_input_ids == 102) | (aug_input_ids == 0)
        augmentable_mask = ~special_tokens_mask & (aug_attention_mask == 1)
        
        # 1. Random masking: 20% probability (HARDER than before)
        mask_prob = torch.rand(batch_size, seq_len, device=aug_input_ids.device)
        should_mask = (mask_prob < self.augment_prob) & augmentable_mask
        aug_input_ids[should_mask] = self.mask_token_id
        
        # 2. Random deletion: 15% probability (HARDER than before)
        delete_prob = torch.rand(batch_size, seq_len, device=aug_input_ids.device)
        should_delete = (delete_prob < self.augment_prob * 0.75) & augmentable_mask & ~should_mask
        aug_attention_mask[should_delete] = 0
        
        # 3. Random replacement: 10% probability (NEW!)
        replace_prob = torch.rand(batch_size, seq_len, device=aug_input_ids.device)
        should_replace = (replace_prob < 0.10) & augmentable_mask & ~should_mask & ~should_delete
        
        # Replace with random tokens from common vocabulary
        random_tokens = torch.randint(
            low=1000,  # Start from common tokens
            high=min(20000, self.vocab_size),
            size=(batch_size, seq_len),
            device=aug_input_ids.device
        )
        aug_input_ids[should_replace] = random_tokens[should_replace]
        
        # Squeeze back to 1D if input was 1D
        if squeeze_output:
            aug_input_ids = aug_input_ids.squeeze(0)
            aug_attention_mask = aug_attention_mask.squeeze(0)
        
        return aug_input_ids, aug_attention_mask


def main_training_pipeline():
    """
    Main training function with pretraining + hard augmentation
    """
    
    print("="*80)
    print("ENHANCED TRAINING PIPELINE")
    print("Step 1: Extended Pretraining (MLM + Topics)")
    print("Step 2: Fine-tuning with Hard Augmentation")
    print("="*80)
    
    # ==================== CONFIGURATION ====================
    CONFIG = {
        'pretrained_model': './pretrained_finbert',  # After pretraining
        'base_model': 'ProsusAI/finbert',  # If no pretraining done
        'use_pretrained': True,  # Set to False to skip pretraining
        'max_length': 128,
        'batch_size': 16,
        'epochs': 7,  # Increased from 5 to 7
        'learning_rate': 2e-5,
        'test_size': 0.2,
        'augment_prob': 0.20,  # HARDER: increased from 0.15 to 0.20
    }
    
    # Paths
    TRAIN_PATH = '../preprocessed_data/train_normalized.csv'  # Use normalized data!
    TEST_PATH = '../preprocessed_data/test_normalized.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}\n")
    
    # ==================== STEP 1: EXTENDED PRETRAINING ====================
    if CONFIG['use_pretrained']:
        print("\n[STEP 1] Checking for pretrained model...")
        import os
        if not os.path.exists(CONFIG['pretrained_model']):
            print("  ⚠ Pretrained model not found. Running pretraining first...")
            print("  This will take ~15-30 minutes on GPU")
            
            # Run pretraining
            from preprocessing.extended_pretraining import run_pretraining
            
            df_full = pd.read_csv(TRAIN_PATH)
            run_pretraining(
                train_texts=df_full['text_normalized'].values,
                model_name=CONFIG['base_model'],
                output_path=CONFIG['pretrained_model'],
                num_epochs=3,
                batch_size=32,
                learning_rate=5e-5,
                mask_prob=0.30
            )
            print("\n  ✓ Pretraining complete!")
        else:
            print(f"  ✓ Found pretrained model at {CONFIG['pretrained_model']}")
        
        model_name = CONFIG['pretrained_model']
    else:
        print("\n[STEP 1] Skipping pretraining, using base model")
        model_name = CONFIG['base_model']
    
    # ==================== STEP 2: LOAD DATA ====================
    print("\n[STEP 2] Loading normalized data...")
    df = pd.read_csv(TRAIN_PATH)
    label_map = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
    print(f"  Dataset: {len(df)} samples")
    for label, count in df['label'].value_counts().sort_index().items():
        print(f"    {label_map[label]}: {count} ({count/len(df)*100:.1f}%)")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text_normalized'].values,  # Use NORMALIZED text!
        df['label'].values,
        test_size=CONFIG['test_size'],
        random_state=42,
        stratify=df['label']
    )
    
    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"\n  Class weights: {', '.join([f'{label_map[i]}: {w:.3f}' for i, w in enumerate(class_weights)])}")
    
    # ==================== STEP 3: INITIALIZE MODEL ====================
    print("\n[STEP 3] Initializing model with harder augmentation...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)
    print(f"  ✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Initialize HARDER augmenter
    augmenter = HarderGPUAugmenter(
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=tokenizer.vocab_size,
        augment_prob=CONFIG['augment_prob']
    )
    print(f"  ✓ HARDER GPU Augmenter: {CONFIG['augment_prob']*100}% prob + replacements (no skip)")
    
    # Create datasets
    class TweetDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128, augmenter=None, is_train=False):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.augmenter = augmenter
            self.is_train = is_train
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
            
            # Apply HARDER augmentation during training
            if self.is_train and self.augmenter is not None:
                input_ids, attention_mask = self.augmenter.augment(input_ids, attention_mask)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer, CONFIG['max_length'],
                                 augmenter=augmenter, is_train=True)
    val_dataset = TweetDataset(val_texts, val_labels, tokenizer, CONFIG['max_length'],
                               augmenter=None, is_train=False)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    
    # ==================== STEP 4: TRAINING ====================
    print("\n[STEP 4] Training with hard augmentation...")
    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    total_steps = len(train_loader) * CONFIG['epochs']
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        return total_loss / len(dataloader)
    
    def eval_model(model, dataloader, loss_fn, device):
        model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        return total_loss / len(dataloader), accuracy_score(true_labels, predictions), predictions, true_labels
    
    # Training loop
    best_val_acc = 0
    for epoch in range(CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device)
        val_loss, val_accuracy, val_preds, val_labels = eval_model(model, val_loader, loss_fn, device)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            # Save best model
            model.save_pretrained('./best_model')
            tokenizer.save_pretrained('./best_model')
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
        if val_accuracy == best_val_acc:
            print(f"  ✨ New best validation accuracy!")
    
    print(f"\n✓ Best validation accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # ==================== STEP 5: EVALUATION ====================
    print("\n[STEP 5] Final evaluation...")
    _, _, final_preds, final_labels = eval_model(model, val_loader, loss_fn, device)
    accuracy = accuracy_score(final_labels, final_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        final_labels, final_preds, average=None, labels=[0,1,2]
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    for i in range(3):
        print(f"{label_map[i]:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print("\n" + classification_report(final_labels, final_preds, target_names=[label_map[i] for i in range(3)]))
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n🎯 Results Summary:")
    print(f"   Pretraining: {'Used' if CONFIG['use_pretrained'] else 'Skipped'}")
    print(f"   Augmentation: HARDER (20% prob + replacements)")
    print(f"   Normalization: Enabled (tickers, prices, percentages)")
    print(f"   Best Val Accuracy: {best_val_acc:.4f}")
    print(f"   Final Accuracy: {accuracy:.4f}")
    print(f"\n💡 Target: 89-93% accuracy")
    print(f"   Achieved: {'✅ YES!' if accuracy >= 0.89 else '⏳ Close! Try more epochs or harder augmentation'}")
    print("="*80)


if __name__ == '__main__':
    main_training_pipeline()
