#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def create_dummy_point_cloud(num_points=4096):
    """Create a random point cloud within a unit sphere."""
    # Generate random points in a cube
    points = np.random.uniform(-1, 1, (num_points, 3))
    
    # Normalize points to lie within a unit sphere
    norms = np.linalg.norm(points, axis=1)
    mask = norms <= 1.0
    
    # If not enough points within the sphere, generate more
    while np.sum(mask) < num_points:
        new_points = np.random.uniform(-1, 1, (num_points, 3))
        new_norms = np.linalg.norm(new_points, axis=1)
        new_mask = new_norms <= 1.0
        points = np.vstack([points, new_points])
        mask = np.concatenate([mask, new_mask])
    
    # Take the first num_points that are within the sphere
    points = points[mask][:num_points]
    
    # Add random normals
    normals = np.random.uniform(-1, 1, (num_points, 3))
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    
    # Combine points and normals
    point_cloud = np.hstack([points, normals])
    
    return point_cloud.astype(np.float32)

def create_dummy_image():
    """Create a random RGB image."""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img)

def create_dummy_dataset(output_dir, num_samples=10, split='train'):
    """Create a dummy dataset with point clouds and images."""
    # Create output directories
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    
    print(f"Creating {split} set with {num_samples} samples in {split_dir}")
    
    for i in tqdm(range(num_samples), desc=f"Creating {split} samples"):
        # Create point cloud
        pc = create_dummy_point_cloud()
        
        # Create image
        img = create_dummy_image()
        
        # Save files
        base_name = f"sample_{i:04d}"
        np.save(os.path.join(split_dir, f"{base_name}_pc.npy"), pc)
        img.save(os.path.join(split_dir, f"{base_name}_img.png"))

def main():
    # Output directory for the dummy dataset
    output_dir = "./dummy_dataset"
    
    # Remove existing directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create train, val, and test splits
    create_dummy_dataset(output_dir, num_samples=20, split='train')
    create_dummy_dataset(output_dir, num_samples=5, split='val')
    create_dummy_dataset(output_dir, num_samples=5, split='test')
    
    print(f"\nDummy dataset created at: {os.path.abspath(output_dir)}")
    print("\nDirectory structure:")
    print(f"{output_dir}/")
    print("├── train/")
    print("│   ├── sample_0000_pc.npy")
    print("│   ├── sample_0000_img.png")
    print("│   └── ...")
    print("├── val/")
    print("└── test/")

if __name__ == "__main__":
    main()
