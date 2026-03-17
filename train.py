import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
from model import VQVAE
from torchvision.utils import save_image

# Default Configuration
DEFAULT_BATCH_SIZE = 1024
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_NUM_EPOCHS = 60
DEFAULT_EMBEDDING_DIM = 8
DEFAULT_NUM_EMBEDDINGS = 8
DEFAULT_DATA_DIR = "./data"
DEFAULT_CHECKPOINT_DIR = "./checkpoints"

# Detect device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"CUDA Available! Devices found: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("Using CPU")

import matplotlib.pyplot as plt

def plot_losses(all_losses, save_path="results/training_curves.png"):
    """Plot training and validation losses"""
    os.makedirs("results", exist_ok=True)
    
    train_total = [l['total_loss'] for l in all_losses['train']]
    train_recon = [l['recon_loss'] for l in all_losses['train']]
    train_commit = [l['commit_loss'] for l in all_losses['train']]
    
    val_total = [l['total_loss'] for l in all_losses['val']]
    val_recon = [l['recon_loss'] for l in all_losses['val']]
    val_commit = [l['commit_loss'] for l in all_losses['val']]
    
    epochs = range(1, len(train_total) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, train, val, title in zip(
        axes,
        [train_total, train_recon, train_commit],
        [val_total,   val_recon,   val_commit],
        ['Total Loss', 'Reconstruction Loss', 'Commitment Loss']
    ):
        ax.plot(epochs, train, label='Train', linewidth=2)
        ax.plot(epochs, val,   label='Val',   linewidth=2, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('VQ-VAE Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curves saved to {save_path}")

def setup_data(data_dir, batch_size):
    """Setup data loaders"""
    # Transform: Convert to tensor and scale to [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    # Load MNIST training data
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        # Dedicate 16 threads specifically to data loading (8 per GPU).
        # This leaves 32 threads completely free for the OS and background tasks.
        num_workers=16,              
        
        # Keep this True. It locks the memory pages so the CPU can 
        # blast data directly into the 4090s' VRAM via PCIe.
        pin_memory=True,            
        
        # Keeps the 16 workers alive between epochs so they don't have to restart.
        persistent_workers=True,
        
        # Tells each of the 16 workers to fetch 4 batches in advance.
        prefetch_factor=4           
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_commit_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (x, _) in enumerate(pbar):
        x = x.to(device)
        
        # Forward pass
        x_hat, indices, commit_loss = model(x)
        
        # Compute loss
        loss_dict = model.compute_loss(x, x_hat, commit_loss)
        total_loss_batch = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss_dict['total_loss'].item()
        total_recon_loss += loss_dict['recon_loss'].item()
        total_commit_loss += loss_dict['commit_loss'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'total': total_loss / (batch_idx + 1),
            'recon': total_recon_loss / (batch_idx + 1),
            'commit': total_commit_loss / (batch_idx + 1)
        })
    
    return {
        'total_loss': total_loss / len(train_loader),
        'recon_loss': total_recon_loss / len(train_loader),
        'commit_loss': total_commit_loss / len(train_loader)
    }


def validate(model, val_loader, device, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_commit_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            
            # Forward pass
            x_hat, indices, commit_loss = model(x)
            
            # Compute loss
            loss_dict = model.compute_loss(x, x_hat, commit_loss)
            
            # Accumulate losses
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_commit_loss += loss_dict['commit_loss'].item()
            
            # Update progress bar
            pbar.set_postfix({
                'total': total_loss / (batch_idx + 1),
                'recon': total_recon_loss / (batch_idx + 1),
                'commit': total_commit_loss / (batch_idx + 1)
            })
            if batch_idx == len(val_loader) - 1:
                os.makedirs("results", exist_ok=True)
                comparison = torch.cat([x[:8], x_hat[:8]])
                save_image(comparison.cpu(), f"results/val_recon_epoch_{epoch+1}.png", nrow=8)
    
    return {
        'total_loss': total_loss / len(val_loader),
        'recon_loss': total_recon_loss / len(val_loader),
        'commit_loss': total_commit_loss / len(val_loader)
    }


def save_checkpoint(model, optimizer, epoch, losses, filepath):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, device):
    """Load checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    print(f"Checkpoint loaded from epoch {epoch}")
    return epoch, losses


def train(num_epochs, batch_size, learning_rate, embedding_dim, num_embeddings, 
          data_dir, checkpoint_dir, resume_from_checkpoint=None):
    """Main training function"""
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup data loaders
    train_loader, val_loader = setup_data(data_dir, batch_size)
    
    # Initialize model
    model = VQVAE(
        embedding_dim=embedding_dim,
        num_embeddings=num_embeddings
    )
        
    model = model.to(DEVICE)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 0
    all_losses = {'train': [], 'val': []}
    
    # Resume from checkpoint if provided
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        start_epoch, all_losses = load_checkpoint(model, optimizer, resume_from_checkpoint, DEVICE)
        start_epoch += 1
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, DEVICE)
        all_losses['train'].append(train_losses)
        
        # Validate
        val_losses = validate(model, val_loader, DEVICE, epoch)
        all_losses['val'].append(val_losses)
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Total: {train_losses['total_loss']:.6f}, "
              f"Recon: {train_losses['recon_loss']:.6f}, "
              f"Commit: {train_losses['commit_loss']:.6f}")
        print(f"  Val   - Total: {val_losses['total_loss']:.6f}, "
              f"Recon: {val_losses['recon_loss']:.6f}, "
              f"Commit: {val_losses['commit_loss']:.6f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"vqvae_epoch_{epoch + 1}.pt")
            save_checkpoint(model, optimizer, epoch, all_losses, checkpoint_path)
    
    # Save final model
    final_path = os.path.join(checkpoint_dir, "vqvae_final.pt")
    save_checkpoint(model, optimizer, num_epochs - 1, all_losses, final_path)
    print(f"\n✅ Training completed! Final model saved to {final_path}")
    

    plot_losses(all_losses)

    return model, all_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VQ-VAE on MNIST")
    
    # Model arguments
    parser.add_argument('--embedding-dim', type=int, default=DEFAULT_EMBEDDING_DIM,
                        help=f'Embedding dimension (default: {DEFAULT_EMBEDDING_DIM})')
    parser.add_argument('--num-embeddings', type=int, default=DEFAULT_NUM_EMBEDDINGS,
                        help=f'Number of codebook embeddings (default: {DEFAULT_NUM_EMBEDDINGS})')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS,
                        help=f'Number of training epochs (default: {DEFAULT_NUM_EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE,
                        help=f'Learning rate (default: {DEFAULT_LEARNING_RATE})')
    
    # Data and checkpoint paths
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Directory to store MNIST data (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR,
                        help=f'Directory to store checkpoints (default: {DEFAULT_CHECKPOINT_DIR})')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    print("="*60)
    print("VQ-VAE Training Configuration")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Embedding Dim: {args.embedding_dim}")
    print(f"Num Embeddings: {args.num_embeddings}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print("="*60 + "\n")
    
    model, losses = train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        num_embeddings=args.num_embeddings,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume_from_checkpoint=args.resume
    )
