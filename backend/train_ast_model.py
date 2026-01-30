"""
Train Audio Spectrogram Transformer (AST) for respiratory sound classification
Optimized for ICBHI 2017 dataset with transfer learning from AudioSet

Usage:
    python train_ast_model.py --data_dir ./data/icbhi --epochs 50 --batch_size 16
"""

import os
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ASTFeatureExtractor, ASTModel
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_utils import (
    download_icbhi_dataset,
    load_icbhi_metadata,
    extract_features_ast,
    spec_augment,
    mixup_data,
    get_class_weights,
    train_val_split,
    CLASS_MAPPING
)


class ASTClassifier(nn.Module):
    """
    Audio Spectrogram Transformer for respiratory sound classification
    Uses pre-trained AST from HuggingFace and fine-tunes for 4-class classification
    """
    
    def __init__(self, num_classes=4, pretrained=True, freeze_backbone_epochs=10):
        super().__init__()
        
        # Load AST backbone with interpolation
        self.load_backbone(pretrained)
        
        self.freeze_backbone_epochs = freeze_backbone_epochs
        
        # Classification head
        hidden_size = self.ast.config.hidden_size  # 768
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initially freeze backbone
        self.freeze_backbone()
        
    def freeze_backbone(self):
        """Freeze AST backbone for initial training"""
        for param in self.ast.parameters():
            param.requires_grad = False
        print("🔒 AST backbone frozen")
        
    def load_backbone(self, pretrained):
        """Load AST backbone with position embedding interpolation (1024->512 frames)"""
        pretrained_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        try:
            # Load config and adjust for 5s input (512 frames)
            from transformers import ASTConfig, ASTModel
            config = ASTConfig.from_pretrained(pretrained_name)
            config.max_length = 512
            
            # Create model with new config (random weights initially)
            self.ast = ASTModel(config)
            
            if pretrained:
                # Load pretrained weights
                old_model = ASTModel.from_pretrained(pretrained_name, ignore_mismatched_sizes=True)
                old_state = old_model.state_dict()
                new_state = self.ast.state_dict()
                
                # Interpolate position embeddings [1, 1214, 768] -> [1, 602, 768]
                old_pos = old_state['embeddings.position_embeddings']
                new_pos = new_state['embeddings.position_embeddings']
                
                if old_pos.shape != new_pos.shape:
                    print(f"Interpolating pos embeddings: {old_pos.shape} -> {new_pos.shape}")
                    cls_tokens = old_pos[:, :2, :]
                    patch_tokens = old_pos[:, 2:, :]
                    
                    n_freq = 12 # Fixed for AST 128 bins
                    n_time_old = patch_tokens.shape[1] // n_freq
                    n_time_new = (new_pos.shape[1] - 2) // n_freq
                    
                    # [1, T, F, H] -> [1, H, T, F]
                    patch_tokens = patch_tokens.reshape(1, n_time_old, n_freq, 768).permute(0, 3, 1, 2)
                    
                    new_patch_tokens = F.interpolate(
                        patch_tokens, 
                        size=(n_time_new, n_freq), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    
                    # [1, H, T, F] -> [1, T*F, H]
                    new_patch_tokens = new_patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, 768)
                    new_pos_final = torch.cat([cls_tokens, new_patch_tokens], dim=1)
                    
                    old_state['embeddings.position_embeddings'] = new_pos_final
                
                # Load modified state dict
                msg = self.ast.load_state_dict(old_state, strict=False)
                print(f"✓ Loaded & Interpolated AST (1024 -> 512 frames)")
            else:
                print("✓ Loaded AST architecture (512 frames, random weights)")
                
        except Exception as e:
            print(f"⚠ Failed to load/interpolate AST: {e}")
            from transformers import ASTConfig, ASTModel
            config = ASTConfig()
            self.ast = ASTModel(config)
        
    def unfreeze_backbone(self):
        """Unfreeze AST backbone for fine-tuning"""
        for param in self.ast.parameters():
            param.requires_grad = True
        print("🔓 AST backbone unfrozen for fine-tuning")
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Mel spectrogram (batch, freq, time)
        """
        # Pass through AST
        # Input shape: (batch, freq, time) or (batch, time, freq)
        # ASTModel automatically handles unsqueezing to (batch, 1, ...)
        outputs = self.ast(x)
        
        # Get pooled output (CLS token)
        pooled_output = outputs.pooler_output  # (batch, hidden_size)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class RespiratoryDataset(Dataset):
    """Dataset for respiratory sound classification"""
    
    def __init__(self, metadata, data_dir, mode='train', augment=True):
        """
        Args:
            metadata: DataFrame with file info and labels
            data_dir: Root directory with audio files
            mode: 'train' or 'val'
            augment: Whether to apply augmentation
        """
        self.metadata = metadata
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.augment = augment and (mode == 'train')
        
        # Create label mapping
        self.label_to_idx = CLASS_MAPPING
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load audio file
        audio_file = self.data_dir / f"{row['filename']}.wav"
        
        if not audio_file.exists():
            # Return zeros if file not found
            features = np.zeros((512, 128), dtype=np.float32)
        else:
            # Extract features
            features = extract_features_ast(audio_file)
            
            # Apply SpecAugment during training
            if self.augment:
                features = spec_augment(features)
        
        # Get label
        label = self.label_to_idx[row['class']]
        
        # Convert to tensors
        features = torch.from_numpy(features)
        label = torch.tensor(label, dtype=torch.long)
        
        return features, label


def train_epoch(model, dataloader, criterion, optimizer, device, mixup=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        # Apply mixup (50% of batches)
        if mixup and np.random.rand() > 0.5 and len(features) > 1:
            # Get random pair for mixing
            indices = torch.randperm(len(features))
            features2 = features[indices]
            labels2 = labels[indices]
            
            # Convert labels to one-hot
            labels_onehot = F.one_hot(labels, num_classes=4).float()
            labels2_onehot = F.one_hot(labels2, num_classes=4).float()
            
            # Mix
            lam = np.random.beta(0.4, 0.4)
            features = lam * features + (1 - lam) * features2
            labels_mixed = lam * labels_onehot + (1 - lam) * labels2_onehot
            
            # Forward pass
            outputs = model(features)
            loss = -(labels_mixed * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        else:
            # Normal forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in tqdm(dataloader, desc='Validation'):
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✓ Confusion matrix saved to {save_path}")


def main(args):
    print("=" * 70)
    print("RevoScope AST Training Pipeline")
    print("=" * 70)
    
    # Setup device
    try:
        # DirectML unstable on RX 9070 XT. Fallback to CPU (Optomized)
        # import torch_directml
        # device = torch_directml.device()
        # print(f"\n🎮 Using device: DirectML (AMD RX 9070 XT)")
        raise ImportError("DirectML disabled for stability")
    except ImportError as e:
        # print(f"⚠ DirectML not found: {e}")
        if torch.backends.mps.is_available():
            device = torch.device('mps')
            print(f"\n📱 Using device: MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"\n🎮 Using device: CUDA (NVIDIA)")
        else:
            device = torch.device('cpu')
            print(f"\n💻 Using device: CPU (no GPU acceleration)")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download/setup dataset
    print("\n" + "=" * 70)
    print("Step 1: Dataset Preparation")
    print("=" * 70)
    
    data_path = download_icbhi_dataset(args.data_dir)
    
    # Load metadata
    print("\nLoading metadata...")
    metadata = load_icbhi_metadata(args.data_dir)
    print(f"✓ Loaded {len(metadata)} samples")
    
    # Show class distribution
    print("\nClass distribution:")
    class_counts = metadata['class'].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} ({count/len(metadata)*100:.1f}%)")
    
    # Split dataset
    train_df, val_df = train_val_split(metadata, val_ratio=args.val_ratio)
    print(f"\n✓ Train: {len(train_df)} | Validation: {len(val_df)}")
    
    # Create datasets
    print("\n" + "=" * 70)
    print("Step 2: Creating Dataloaders")
    print("=" * 70)
    
    train_dataset = RespiratoryDataset(train_df, args.data_dir, mode='train', augment=True)
    val_dataset = RespiratoryDataset(val_df, args.data_dir, mode='val', augment=False)
    
    # Calculate class weights for balanced sampling
    train_labels = [CLASS_MAPPING[cls] for cls in train_df['class']]
    class_weights = get_class_weights(train_labels)
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True
    )
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("Step 3: Model Initialization")
    print("=" * 70)
    
    model = ASTClassifier(num_classes=4, pretrained=True, freeze_backbone_epochs=args.freeze_epochs)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n" + "=" * 70)
    print("Step 4: Training")
    print("=" * 70)
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 70)
        
        # Unfreeze backbone after specified epochs
        if epoch == args.freeze_epochs:
            model.unfreeze_backbone()
            # Reduce learning rate for fine-tuning
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr * 0.1
            print(f"🔓 Backbone unfrozen, LR reduced to {args.lr * 0.1}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, mixup=args.mixup
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'class_mapping': CLASS_MAPPING
            }
            
            save_path = output_dir / 'respiratory_ast_best.pth'
            torch.save(checkpoint, save_path)
            print(f"✓ Best model saved (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠ Early stopping triggered (patience: {args.patience})")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Step 5: Final Evaluation")
    print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'respiratory_ast_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    val_loss, val_acc, val_preds, val_labels = validate(
        model, val_loader, criterion, device
    )
    
    print(f"\n🎯 Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Classification report
    class_names = ['Normal', 'Crackle', 'Wheeze', 'Both']
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Save confusion matrix
    plot_confusion_matrix(
        val_labels, val_preds, class_names,
        output_dir / 'confusion_matrix.png'
    )
    
    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History - Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training History - Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png')
    plt.close()
    print(f"✓ Training history saved")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\nModel saved to: {output_dir / 'respiratory_ast_best.pth'}")
    print(f"Final accuracy: {best_val_acc*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AST for respiratory classification")
    
    # Dataset
    parser.add_argument('--data_dir', type=str, default='./data/icbhi',
                        help='Path to ICBHI dataset')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (Optimized for CPU)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--freeze_epochs', type=int, default=10,
                        help='Number of epochs to freeze backbone')
    parser.add_argument('--mixup', action='store_true', default=True,
                        help='Use mixup augmentation')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # System
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='./weights',
                        help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    main(args)
