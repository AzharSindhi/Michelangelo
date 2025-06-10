
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
from michelangelo.models.asl_diffusion.clip_asl_diffuser_pl_module import ClipASLDiffuser
from michelangelo.callbacks.pointcloud_diff_sample import PointCloudSampler


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
    config_path = "configs/image_cond_diffuser_asl/image-ASLDM-256.yaml"
    config = OmegaConf.load(config_path)

    # run name based on time and process id and prefix
    # Initialize model
    clip_diffuser = ClipASLDiffuser(**config.model.params)

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
    vis_dir = f'out_sampled_pcs/{timenow}'
    print(f"INFO: visualization directory:{vis_dir}")
    # Point cloud sampler callback
    pc_sampler = PointCloudSampler(
        save_dir=vis_dir,
        max_samples=4,  # Number of samples to save per epoch
        every_n_epochs=1  # Save every epoch
    )
    # bs_finder = BatchSizeFinder(mode='binsearch', init_val=config.batch_size)
    callbacks = [pc_sampler]
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
        model=clip_diffuser,
        datamodule=datamodule,
        ckpt_path=ckpt_path
    )

    print("------INFO----------------")
    print("CD Test:", pc_sampler.cd_all_test / len(datamodule.predict_dataloader()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt_path", "-c",type=str)
    # args = parser.parse_args()
    # args.run_name = args.run_name + "_dino"
    ckpt_path = "diffusion_checkpoints/dino_nocontrast_diff_bfebcd4d694543bebdae39244b78b78b/last.ckpt"
    evalulate(ckpt_path)