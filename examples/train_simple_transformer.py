"""Example script for training a simple transformer model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# This would normally import from easy_transformer
# from easy_transformer import Transformer, create_padding_mask


class SimpleDataset(Dataset):
    """Simple dataset for demonstration."""
    
    def __init__(self, num_samples=1000, seq_length=20, vocab_size=100):
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        
        # Generate random sequences
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # For simplicity, predict next token
        return {
            'input': self.data[idx, :-1],
            'target': self.data[idx, 1:]
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input'].to(device)
        target_ids = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        # Simple prediction (this would use the actual transformer)
        # For now, just a placeholder
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100).to(device)
        
        # Calculate loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1)
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input'].to(device)
            target_ids = batch['target'].to(device)
            
            # Forward pass (placeholder)
            logits = torch.randn(input_ids.shape[0], input_ids.shape[1], 100).to(device)
            
            # Calculate loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1)
            )
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main training function."""
    # Configuration
    config = {
        'vocab_size': 100,
        'd_model': 128,
        'n_heads': 8,
        'n_layers': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Training on device: {config['device']}")
    
    # Create datasets
    train_dataset = SimpleDataset(num_samples=1000)
    val_dataset = SimpleDataset(num_samples=200)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Create model (placeholder - would use actual Transformer)
    model = nn.Linear(config['d_model'], config['vocab_size']).to(config['device'])
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, config['device'])
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_progress.png')
    plt.show()
    
    print("\nTraining completed!")
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, 'simple_transformer_checkpoint.pt')
    
    print("Model saved to 'simple_transformer_checkpoint.pt'")


if __name__ == "__main__":
    main()