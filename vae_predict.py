
import os
import torch
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Dict, Any, List, Tuple
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.backends import cudnn
from michelangelo.models.tsal.asl_pl_module import AlignedShapeAsLatentPLModule
from michelangelo.callbacks.pointcloud_write import PointCloudSaver

from pytorch_lightning.loggers import MLFlowLogger
from michelangelo.data.dataset import get_dataset
from tqdm import tqdm
from pytorch_lightning.tuner import lr_finder
import time
import argparse


class ShapeNetViPCDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 4, num_workers: int = 4,
                 view_align: bool = True, category: str = "plane", mini: bool = True,
                 image_size: int = 224):
        super().__init__()
        self.data_dir = os.path.expanduser(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.view_align = view_align
        self.category = category
        self.mini = mini
        self.image_size = image_size

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = get_dataset(self.data_dir, phase="train", view_align=self.view_align, category=self.category, mini=self.mini, image_size=self.image_size)
        self.val_dataset = get_dataset(self.data_dir, phase="test", view_align=self.view_align, category=self.category, mini=self.mini, image_size=self.image_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=self.num_workers > 0
        )


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def evalulate(ckpt_path):
    # seed everything
    set_seed(42)
    # Load config
    config_path = "configs/aligned_shape_latents/shapevae-256.yaml"
    config = OmegaConf.load(config_path)

    # Initialize model
    ignore_keys = () 
    sita_vae = AlignedShapeAsLatentPLModule(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        shape_module_cfg=config.model.params.shape_module_cfg,
        aligned_module_cfg=config.model.params.aligned_module_cfg,
        loss_cfg=config.model.params.loss_cfg,
        optimizer_cfg=config.model.params.optimizer_cfg,
        ckpt_path=ckpt_path,
        ignore_keys=ignore_keys
    )

    # Setup data
    datamodule = ShapeNetViPCDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.batch_size,
        num_workers=config.data.num_workers,
        view_align=config.data.view_align,
        category=config.data.category,
        mini=config.data.mini,
        image_size=config.data.image_size
    )

    timenow = time.strftime('%Y%m%d-%H%M')
    print(f"INFO: Visualization directory: out_pointclouds/{timenow}")

    # Point cloud saver callback
    pc_saver = PointCloudSaver(
        save_dir=f"out_pointclouds/{timenow}",
        max_samples=2,  # Number of samples to save per epoch
        every_n_epochs=1
    )
    
    callbacks = [pc_saver]
    
    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=config.devices,
        logger=None,
        # strategy=config.strategy,
        enable_model_summary=True,
        callbacks=callbacks
    )

    # Train the model
    trainer.predict(
        model=sita_vae,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )
    print("here")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt_path", "-c",type=str)
    # args = parser.parse_args()
    # args.run_name = args.run_name + "_dino"
    ckpt_path = "checkpoints_runs/vae_nocontrast_dino_20250608-0024_d94dbf748d754c0e849c4621086fa69f/sita_vae-best.ckpt"
    evalulate(ckpt_path)