import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

# Import our models
from michelangelo.models.end2end import End2End

# Set random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Configuration
class Config:
    # Model
    num_latents = 256
    embed_dim = 64
    point_feats = 3  # XYZ coordinates
    clip_model_name = "openai/clip-vit-large-patch14"
    
    # Diffusion
    diffusion_steps = 1000
    diffusion_width = 768
    diffusion_layers = 6
    diffusion_heads = 12
    
    # Training
    batch_size = 8
    num_workers = 4
    learning_rate = 1e-4
    num_epochs = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paths
    data_dir = "data/shapenet"  # Update this to your dataset path
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)


# Dummy dataset for demonstration
class PointCloudDataset(Dataset):
    def __init__(self, num_samples=1000, num_points=2048):
        self.num_samples = num_samples
        self.num_points = num_points
        
        # Generate random point clouds for demonstration
        # In practice, load from your dataset
        self.point_clouds = [
            torch.randn(num_points, 6)  # XYZ + normals
            for _ in range(num_samples)
        ]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return point cloud and dummy image (for demonstration)
        point_cloud = self.point_clouds[idx]
        image = torch.randn(3, 224, 224)  # Dummy image
        
        return {
            "point_cloud": point_cloud.float(),
            "image": image.float()
        }


def train():
    # Initialize config
    cfg = Config()
    
    # Create model
    model = End2End().to(cfg.device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.99),
        weight_decay=1e-2
    )
    
    # Create datasets and dataloaders
    train_dataset = PointCloudDataset(num_samples=1000)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Training loop
    model.train()
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        
        for batch in pbar:
            # Move data to device
            point_cloud = batch["point_cloud"].to(cfg.device)
            image = batch["image"].to(cfg.device)
            text = "I am a dummy text"
            # text as batch
            # text = torch.repeat_interleave(text, cfg.batch_size, dim=0)
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                surface=point_cloud,
                images=image,
                text=text
            )
            
            # Backward pass
            loss = outputs["loss"]
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        # Print epoch statistics
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{cfg.num_epochs}, Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(cfg.save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    print("Training complete!")


def sample(model_path: str, num_samples: int = 4):
    """
    Sample from a trained model
    """
    cfg = Config()
    
    # Initialize model
    model = EndToEndModel(
        num_latents=cfg.num_latents,
        embed_dim=cfg.embed_dim,
        point_feats=cfg.point_feats,
        clip_model_name=cfg.clip_model_name,
        diffusion_width=cfg.diffusion_width,
        diffusion_layers=cfg.diffusion_layers,
        diffusion_heads=cfg.diffusion_heads,
        diffusion_steps=cfg.diffusion_steps,
        device=cfg.device
    ).to(cfg.device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=cfg.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Generate random condition (in practice, use CLIP embeddings from image/text)
    condition = torch.randn(num_samples, cfg.embed_dim * 2, device=cfg.device)
    
    # Generate samples
    with torch.no_grad():
        outputs = model.sample(
            condition=condition,
            shape=(num_samples, cfg.num_latents, cfg.embed_dim)
        )
    
    # Outputs contains 'latents' and 'decoded' point clouds
    print(f"Generated {num_samples} samples!")
    return outputs


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint for sampling")
    parser.add_argument("--num_samples", type=int, default=4, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train()
    elif args.mode == "sample":
        if not args.checkpoint:
            raise ValueError("Please provide a checkpoint path with --checkpoint")
        sample(args.checkpoint, args.num_samples)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
