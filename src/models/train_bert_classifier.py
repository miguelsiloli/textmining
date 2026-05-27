"""
BERT Sentiment Classification Training Script

This script trains a BERT transformer model for tweet sentiment classification.
Handles class imbalance and computes comprehensive evaluation metrics.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class TweetDataset(Dataset):
    """Custom Dataset for tweet sentiment classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier:
    """BERT-based sentiment classifier with class imbalance handling"""
    
    def __init__(self, model_name='bert-base-uncased', num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        self.label_map = {0: 'Bearish', 1: 'Bullish', 2: 'Neutral'}
        self.history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    def prepare_data(self, train_path, test_size=0.2, max_length=128, batch_size=16):
        """Load and prepare data with train/validation split"""
        print("\n[1/6] Loading and preparing data...")
        
        # Load training data
        df = pd.read_csv(train_path)
        print(f"Total samples: {len(df)}")
        
        # Analyze class distribution
        print("\nClass distribution:")
        for label, count in df['label'].value_counts().sort_index().items():
            print(f"  {self.label_map[label]}: {count} ({count/len(df)*100:.1f}%)")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].values,
            df['label'].values,
            test_size=test_size,
            random_state=42,
            stratify=df['label']  # Maintain class distribution in splits
        )
        
        print(f"\nTrain samples: {len(train_texts)}")
        print(f"Validation samples: {len(val_texts)}")
        
        # Compute class weights for handling imbalance
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        print("\nClass weights (for handling imbalance):")
        for label, weight in enumerate(class_weights):
            print(f"  {self.label_map[label]}: {weight:.3f}")
        
        # Create datasets
        train_dataset = TweetDataset(train_texts, train_labels, self.tokenizer, max_length)
        val_dataset = TweetDataset(val_texts, val_labels, self.tokenizer, max_length)
        
        # Create dataloaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        return len(train_texts), len(val_texts)
    
    def train(self, epochs=3, learning_rate=2e-5):
        """Train the model"""
        print(f"\n[2/6] Training model for {epochs} epochs...")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(self.train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Loss function with class weights
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(self.train_loader, desc="Training")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                loss = loss_fn(outputs.logits, labels)
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(self.train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(self.val_loader, loss_fn)
            
            # Save history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
    
    def evaluate(self, dataloader, loss_fn=None):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                if loss_fn:
                    loss = loss_fn(outputs.logits, labels)
                    total_loss += loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        
        if loss_fn:
            avg_loss = total_loss / len(dataloader)
            return avg_loss, accuracy
        
        return predictions, true_labels
    
    def compute_metrics(self):
        """Compute comprehensive evaluation metrics"""
        print("\n[3/6] Computing evaluation metrics...")
        
        predictions, true_labels = self.evaluate(self.val_loader)
        
        # Compute metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, labels=[0, 1, 2]
        )
        
        # Macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro'
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Print results
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        print("\nPer-Class Metrics:")
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 80)
        for i in range(3):
            print(f"{self.label_map[i]:<15} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
        
        print("\nAveraged Metrics:")
        print(f"{'Metric':<15} {'Macro Avg':<12} {'Weighted Avg':<12}")
        print("-" * 80)
        print(f"{'Precision':<15} {precision_macro:<12.4f} {precision_weighted:<12.4f}")
        print(f"{'Recall':<15} {recall_macro:<12.4f} {recall_weighted:<12.4f}")
        print(f"{'F1-Score':<15} {f1_macro:<12.4f} {f1_weighted:<12.4f}")
        
        # Classification report
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            true_labels, 
            predictions,
            target_names=[self.label_map[i] for i in range(3)]
        ))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def plot_confusion_matrix(self, metrics, output_dir='outputs/figures'):
        """Plot confusion matrix"""
        print("\n[4/6] Generating confusion matrix...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        cm = confusion_matrix(metrics['true_labels'], metrics['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=[self.label_map[i] for i in range(3)],
            yticklabels=[self.label_map[i] for i in range(3)]
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Confusion matrix saved to {save_path}")
    
    def plot_training_history(self, output_dir='outputs/figures'):
        """Plot training history"""
        print("\n[5/6] Plotting training history...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-o', label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-o', label='Val Loss')
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['val_accuracy'], 'g-o', label='Val Accuracy')
        ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training history saved to {save_path}")
    
    def save_model(self, output_dir='models'):
        """Save the trained model"""
        print(f"\n[6/6] Saving model...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        model_path = os.path.join(output_dir, 'bert_sentiment_classifier')
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Save training history
        history_path = os.path.join(output_dir, 'training_history.pkl')
        with open(history_path, 'wb') as f:
            pickle.dump(self.history, f)
        
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Training history saved to {history_path}")

def main():
    """Main training routine"""
    print("="*80)
    print("BERT SENTIMENT CLASSIFICATION TRAINING")
    print("="*80)
    
    # Initialize classifier
    classifier = BERTSentimentClassifier()
    
    # Prepare data
    train_size, val_size = classifier.prepare_data(
        train_path='data/train.csv',
        test_size=0.2,
        max_length=128,
        batch_size=16
    )
    
    # Train model
    classifier.train(epochs=3, learning_rate=2e-5)
    
    # Compute metrics
    metrics = classifier.compute_metrics()
    
    # Generate visualizations
    classifier.plot_confusion_matrix(metrics)
    classifier.plot_training_history()
    
    # Save model
    classifier.save_model()
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
