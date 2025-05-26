#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
from torch.utils.data import DataLoader
from michelangelo.datasets.point_cloud import PointCloudDataset
from michelangelo.data.utils import worker_init_fn, collation_fn

def test_dataloader():
    # Path to the dummy dataset
    data_dir = "./dummy_dataset"
    
    # Create dataset
    dataset = PointCloudDataset(
        data_dir=data_dir,
        split="train",
        num_points=4096,
        num_volume_samples=8192,
        num_near_samples=2048
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        collate_fn=collation_fn,
        pin_memory=True
    )
    
    # Test a few batches
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} ({v.dtype})")
            else:
                print(f"  {k}: {type(v).__name__}")
        
        if i >= 2:  # Only check first 3 batches
            break

if __name__ == "__main__":
    # Add the parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_dataloader()
