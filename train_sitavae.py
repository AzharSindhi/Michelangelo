
import os
import torch
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Dict, Any, List, Tuple
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt

from michelangelo.models.tsal.asl_pl_module import AlignedShapeAsLatentPLModule
from michelangelo.callbacks.pointcloud_write import PointCloudSaver

from pytorch_lightning.loggers import MLFlowLogger
from michelangelo.data.dataset import get_dataset


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
        # Split dataset into train/val
        self.train_dataset = get_dataset(self.data_dir, phase="train", view_align=self.view_align, category=self.category, mini=self.mini, image_size=self.image_size)
        self.val_dataset = get_dataset(self.data_dir, phase="test", view_align=self.view_align, category=self.category, mini=self.mini, image_size=self.image_size)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )


def train():
    # Load config
    config_path = "configs/aligned_shape_latents/shapevae-256.yaml"
    config = OmegaConf.load(config_path)

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./mlruns")

    # Initialize model
    # ignore_keys = ["model.shape_model.encoder.query", "model.shape_model.geo_decoder.output_proj.weight", "model.shape_model.geo_decoder.output_proj.bias"]
    ignore_keys = []
    sita_vae = AlignedShapeAsLatentPLModule(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float32,
        shape_module_cfg=config.model.params.shape_module_cfg,
        aligned_module_cfg=config.model.params.aligned_module_cfg,
        loss_cfg=config.model.params.loss_cfg,
        optimizer_cfg=config.model.params.optimizer_cfg,
        ckpt_path=config.model.params.ckpt_path,
        ignore_keys=ignore_keys
    )

    # state_dict = checkpoint['state_dict']
    # model_state_dict = model.state_dict()

    # # Only load matching shapes
    # for k in model_state_dict:
    #     if k in state_dict and state_dict[k].shape == model_state_dict[k].shape:
    #         model_state_dict[k] = state_dict[k]

    # model.load_state_dict(model_state_dict)

    # Setup data
    datamodule = ShapeNetViPCDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        view_align=config.data.view_align,
        category=config.data.category,
        mini=config.data.mini,
        image_size=config.data.image_size
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath='checkpoints_runs/',
        filename='sita_vae-{epoch:02d}-{val_total_loss:.2f}',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Point cloud saver callback
    pc_saver = PointCloudSaver(
        save_dir='out_pointclouds',
        max_samples=4,  # Number of samples to save per epoch
        every_n_epochs=10  # Save every epoch
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=1000,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor, pc_saver],
        enable_model_summary=True,
        log_every_n_steps=50,
        logger=mlf_logger,
        check_val_every_n_epoch=5,  # Run validation once per 5 epochs
    )

    # Train the model
    trainer.fit(
        model=sita_vae,
        datamodule=datamodule,
        # ckpt_path="path/to/checkpoint.ckpt"  # Uncomment to resume training
    )


if __name__ == "__main__":
    train()