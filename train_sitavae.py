
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
from mlflow_git import log_git_details


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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)

def train(run_name_prefix="", experiment_name="sitavae"):
    # seed everything
    set_seed(42)
    
    # Load config
    config_path = "configs/aligned_shape_latents/shapevae-256.yaml"
    config = OmegaConf.load(config_path)
    if config.overfit_batches > 0:
        config.model.dropout = 0.0

    # run name based on time and process id and prefix
    run_name = f"{run_name_prefix}_{time.strftime('%Y%m%d-%H%M')}"
    mlf_logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
    log_git_details(mlf_logger)
    # return 
    print(f"INFO: Run name: {run_name}, run_id: {mlf_logger.run_id}")
    # Initialize model
    ignore_keys = ["model.shape_model.geo_decoder"] # retraining only point cloud reconstruction
    # ignore_keys = []
    sita_vae = AlignedShapeAsLatentPLModule(
        device=config.device,
        dtype=torch.float32,
        shape_module_cfg=config.model.params.shape_module_cfg,
        aligned_module_cfg=config.model.params.aligned_module_cfg,
        loss_cfg=config.model.params.loss_cfg,
        optimizer_cfg=config.model.params.optimizer_cfg,
        ckpt_path=config.model.params.ckpt_path if config.use_ckpt else None,
        ignore_keys=ignore_keys
    )
    # everything except the geometry decoder is freezed, set trainable
    # for name, param in sita_vae.named_parameters():
    #     if "geo_decoder" not in name:
    #         param.requires_grad = False
    #     else:
    #         print(f"Setting {name} to trainable")
    #         param.requires_grad = True
    
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
        batch_size=config.batch_size,
        num_workers=config.data.num_workers,
        view_align=config.data.view_align,
        category=config.data.category,
        mini=config.data.mini,
        image_size=config.data.image_size
    )
    # train_dataloader = datamodule.train_dataloader()
    # val_dataloader = datamodule.val_dataloader()
    # dataloader sanity check
    # for batch in tqdm(train_dataloader, desc="Training dataloader sanity check"):
    #     pass
    # for batch in tqdm(val_dataloader, desc="Validation dataloader sanity check"):
    #     pass
    
    # Callbacks
    # mlflow run id
    dirpath = f"checkpoints_runs/{run_name}_{mlf_logger.run_id}"
    print(f"INFO: Checkpoints directory: {dirpath}")
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath=dirpath,
        filename='sita_vae-best',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    print(f"INFO: Visualization directory: out_pointclouds/{run_name_prefix}_{mlf_logger.run_id}")

    # Point cloud saver callback
    pc_saver = PointCloudSaver(
        save_dir=f"out_pointclouds/{run_name}_{mlf_logger.run_id}",
        max_samples=2,  # Number of samples to save per epoch
        every_n_epochs=config.check_val_every_n_epoch  # Save every epoch
    )
    
    callbacks = [lr_monitor, pc_saver, checkpoint_callback]
    if config.use_swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
        callbacks.append(swa_callback)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu",
        devices=config.devices,
        strategy=config.strategy,
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=config.log_every_n_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        logger=mlf_logger,
        check_val_every_n_epoch=config.check_val_every_n_epoch,  # Run validation once per 5 epochs
        fast_dev_run=config.fast_dev_run,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        overfit_batches=config.overfit_batches,
    )

    if config.use_lr_finder:
        lr_finder = trainer.tuner.lr_find(sita_vae, max_lr=config.max_lr, datamodule=datamodule, update_attr=False)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder.png")
        sita_vae.hparams.lr = lr_finder.suggestion()
        print(f"Learning rate set to {sita_vae.hparams.lr}")

    # Train the model
    trainer.fit(
        model=sita_vae,
        datamodule=datamodule,
        # ckpt_path="path/to/checkpoint.ckpt"  # Uncomment to resume training
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", "-r",type=str, required=True)
    parser.add_argument("--experiment_name", "-e",type=str, default="sita_vae_dino")
    args = parser.parse_args()
    args.run_name = args.run_name + "_dino"
    train(args.run_name, args.experiment_name)