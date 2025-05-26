#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from michelangelo.data.transforms import build_transforms
from michelangelo.datasets.point_cloud import PointCloudDataModule
from michelangelo.models.asl_diffusion.clip_asl_diffuser_pl_module import ClipASLDiffuser
from michelangelo.utils.misc import instantiate_from_config


def quick_test():
    # Load configuration from YAML file
    config_path = "configs/test_config/image-ASLDM-256.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Add runtime configuration
    runtime_cfg = {
        'trainer': {
            'max_epochs': 1,
            'log_every_n_steps': 1,
            'limit_train_batches': 10,
            'limit_val_batches': 10,
            'limit_test_batches': 10,
            'accelerator': 'gpu',
            'devices': 1,
            'enable_checkpointing': False,
            'logger': True,
            'default_root_dir': './test_logs/quick_test'
        }
    }
    
    # Merge configurations
    cfg = OmegaConf.merge(cfg, OmegaConf.create(runtime_cfg))

    # Create output directory
    os.makedirs(cfg.trainer.default_root_dir, exist_ok=True)

    # Initialize model with learning rate
    model = instantiate_from_config(cfg.model)
    model.learning_rate = 1e-4  # Set learning rate

    # Initialize datamodule
    datamodule = instantiate_from_config(cfg.data)

    # Initialize trainer
    trainer = pl.Trainer(**cfg.trainer)
    # Run quick test
    trainer.fit(model, datamodule=datamodule)
    print("\nQuick test completed successfully!")

if __name__ == "__main__":
    quick_test()
