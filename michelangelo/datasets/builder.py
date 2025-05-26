# -*- coding: utf-8 -*-

from typing import Dict, Any, Optional
import pytorch_lightning as pl
from omegaconf import DictConfig

from .point_cloud import PointCloudDataModule


def build_datamodule(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    **kwargs
) -> pl.LightningDataModule:
    """Build a datamodule for training.
    
    Args:
        data_dir: Path to the dataset directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        **kwargs: Additional arguments for the datamodule
        
    Returns:
        A PyTorch Lightning DataModule
    """
    # For now, we'll use the PointCloudDataModule
    # In the future, we can add support for other dataset types
    return PointCloudDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        **kwargs
    )
