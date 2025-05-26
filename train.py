#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from omegaconf import OmegaConf

from michelangelo.models.asl_diffusion.clip_asl_diffuser_pl_module import ClipASLDiffuser
from michelangelo.datasets import build_datamodule

def parse_args():
    parser = argparse.ArgumentParser(description="Train Michelangelo model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/image_cond_diffuser_asl/image-ASLDM-256.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save logs and checkpoints"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 for all available)"
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=16,
        choices=[16, 32],
        help="Floating point precision"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Initialize model
    model = ClipASLDiffuser(**config.model.params)
    
    # Setup data module
    datamodule = build_datamodule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        **config.pointcloud_data.params
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "checkpoints"),
        filename="{epoch:03d}-{val_loss:.4f}",
        save_top_k=5,
        monitor="val_loss",
        mode="min",
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger(args.log_dir, name="michelangelo")
    
    # Initialize trainer
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        precision=args.precision,
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4 times per epoch
        resume_from_checkpoint=args.resume_from,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )
    
    # Train!
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
