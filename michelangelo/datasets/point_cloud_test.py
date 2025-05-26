# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, List, Dict, Any, Tuple
from omegaconf import DictConfig

from ..data.transforms import build_transforms
from ..data.utils import worker_init_fn, collation_fn


class PointCloudDataset(Dataset):
    """Dataset for point cloud data with optional image conditioning."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        num_points: int = 4096,
        num_volume_samples: int = 1024,
        num_near_samples: int = 1024,
    ):
        """
        Args:
            data_dir: Path to the dataset directory
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to be applied to the data
            num_points: Number of points to sample from each point cloud
            num_volume_samples: Number of volume samples for occupancy prediction
            num_near_samples: Number of near-surface samples for occupancy prediction
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.num_points = num_points
        self.num_volume_samples = num_volume_samples
        self.num_near_samples = num_near_samples
        
        # Load data paths
        self.data_paths = self._load_data_paths()
        
        if len(self.data_paths) == 0:
            raise ValueError(f"No data found in {os.path.join(data_dir, split)}")
    
    def _load_data_paths(self) -> List[Dict[str, str]]:
        """Load paths to data files."""
        data_paths = []
        
        # Look for point cloud files
        pc_pattern = os.path.join(self.data_dir, self.split, "*_pc.npy")
        pc_files = sorted(glob.glob(pc_pattern))
        
        for pc_file in pc_files:
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(pc_file))[0].replace("_pc", "")
            
            # Look for corresponding image file
            img_file = os.path.join(self.data_dir, self.split, f"{base_name}_img.png")
            if not os.path.exists(img_file):
                img_file = None
                
            data_paths.append({
                'pc': pc_file,
                'img': img_file,
                'id': base_name
            })
            
        return data_paths
    
    def __len__(self) -> int:
        return len(self.data_paths)
    
    def _load_point_cloud(self, pc_path: str) -> Dict[str, torch.Tensor]:
        """Load and process a point cloud."""
        # Load point cloud (expected shape: [N, 3] or [N, 6] if normals are included)
        pc_data = np.load(pc_path)
        
        # Ensure we have the right shape
        if pc_data.shape[1] not in [3, 6]:
            pc_data = pc_data.transpose(1, 0)  # Transpose if channels last
            
        # Convert to torch tensor
        pc_data = torch.from_numpy(pc_data).float()
        
        # Sample points if needed
        if pc_data.shape[0] > self.num_points:
            idx = torch.randperm(pc_data.shape[0])[:self.num_points]
            pc_data = pc_data[idx]
        
        # Separate coordinates and normals if available
        if pc_data.shape[1] == 6:
            coords = pc_data[:, :3]
            normals = pc_data[:, 3:]
        else:
            coords = pc_data
            normals = torch.zeros_like(coords)
        
        # Create sample dictionary
        sample = {
            'surface': torch.cat([coords, normals], dim=-1),  # [N, 6]
            'geo_points': torch.randn(self.num_volume_samples, 3),  # Random points in space
            'tex_points': torch.randn(self.num_near_samples, 3),  # Points near surface
        }
        
        return sample
    
    def _load_image(self, img_path: Optional[str]) -> Optional[torch.Tensor]:
        """Load and preprocess an image."""
        if img_path is None or not os.path.exists(img_path):
            return None
            
        # TODO: Add image loading and preprocessing
        # For now, return a dummy tensor
        return torch.zeros(3, 224, 224)  # Dummy image
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset."""
        data = self.data_paths[idx]
        
        # Load point cloud
        sample = self._load_point_cloud(data['pc'])
        
        # Load image if available
        img = self._load_image(data['img'])
        if img is not None:
            sample['image'] = img
        
        # Apply transforms if any
        if self.transform:
            sample = self.transform(sample)
            
        return sample


class PointCloudDataModule(pl.LightningDataModule):
    """Data module for point cloud datasets."""
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        num_points: int = 4096,
        num_volume_samples: int = 1024,
        num_near_samples: int = 1024,
        train_val_test_split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        **kwargs
    ):
        """
        Args:
            data_dir: Path to the dataset directory
            batch_size: Batch size per GPU
            num_workers: Number of data loading workers
            num_points: Number of points to sample from each point cloud
            num_volume_samples: Number of volume samples for occupancy prediction
            num_near_samples: Number of near-surface samples for occupancy prediction
            train_val_test_split: Split ratios for train/val/test
            **kwargs: Additional arguments
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_points = num_points
        self.num_volume_samples = num_volume_samples
        self.num_near_samples = num_near_samples
        self.train_val_test_split = train_val_test_split
        
        # Setup transforms
        self.train_transform = build_transforms(kwargs.get('train_transform', None))
        self.val_transform = build_transforms(kwargs.get('val_transform', None))
        self.test_transform = build_transforms(kwargs.get('test_transform', None))
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = PointCloudDataset(
                data_dir=self.data_dir,
                split="train",
                transform=self.train_transform,
                num_points=self.num_points,
                num_volume_samples=self.num_volume_samples,
                num_near_samples=self.num_near_samples
            )
            
            self.val_dataset = PointCloudDataset(
                data_dir=self.data_dir,
                split="val",
                transform=self.val_transform,
                num_points=self.num_points,
                num_volume_samples=self.num_volume_samples,
                num_near_samples=self.num_near_samples
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = PointCloudDataset(
                data_dir=self.data_dir,
                split="test",
                transform=self.test_transform,
                num_points=self.num_points,
                num_volume_samples=self.num_volume_samples,
                num_near_samples=self.num_near_samples
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collation_fn,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collation_fn,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collation_fn,
            drop_last=False
        )
