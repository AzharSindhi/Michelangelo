
import os
import torch
import numpy as np
import os.path as osp
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, StochasticWeightAveraging, LambdaCallback
from torch.utils.data import DataLoader, Dataset, random_split
from typing import Optional, Dict, Any, List, Tuple
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import random
from torch.backends import cudnn
from michelangelo.models.tsal.asl_pl_module import AlignedShapeAsLatentPLModule
from michelangelo.callbacks.pointcloud_diff_sample import PointCloudSampler

from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from michelangelo.data.dataset import get_dataset
from tqdm import tqdm
from pytorch_lightning.tuner import lr_finder
import time
import argparse
from michelangelo.models.asl_diffusion.clip_asl_diffuser_pl_module import ClipASLDiffuser
from michelangelo.callbacks.logger_callbacks import GitInfoLogger

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
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
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

def train(args):
    # seed everything
    set_seed(42)
    
    # Load config
    config_path = "configs/image_cond_diffuser_asl/image-ASLDM-256.yaml"
    config = OmegaConf.load(config_path)
    if args.use_clip_cond:
        config.model.params.cond_stage_config = config.clip_cond_stage_config
        # config.model.params.aligned_module_cfg.params.target = "michelangelo.models.tsal.clip_asl_module.CLIPAlignedShapeAsLatentModule"
    else:
        config.model.params.cond_stage_config = config.dino_cond_stage_config
        # config.model.params.aligned_module_cfg.params.target = "michelangelo.models.tsal.dino_asl_module.DinoAlignedShapeAsLatentModule"
    # run name based on time and process id and prefix
    run_name = f"diff_{args.run_name}_{time.strftime('%Y%m%d-%H%M')}"
    git_logger = GitInfoLogger()
    commits_info = git_logger.get_git_info()
    git_tags = [f"{key}: {value}" for key, value in commits_info.items()]
    logger = WandbLogger(project=args.experiment_name, name=run_name, tags=git_tags, config=dict(config))
    # for logging git commits
    git_logger.log_git_diff(logger)
    logger.log_hyperparams(config)
    logger.log_hyperparams(vars(args))

    print(f"INFO: Run name: {run_name}")
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

    dirpath = f"diffusion_checkpoints/{run_name}_{logger.experiment.id}"
    print(f"INFO: Save direcotry:{dirpath}")
    checkpoint_callback = ModelCheckpoint(
        monitor='val/total_loss',
        dirpath=dirpath,
        filename='clip_diffuser-best',
        save_top_k=1,
        save_last=True,
        mode='min',
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    vis_dir = f'out_sampled_pcs/{run_name}_{logger.experiment.id}'
    print(f"INFO: visualization directory:{vis_dir}")
    # Point cloud sampler callback
    pc_sampler = PointCloudSampler(
        save_dir=vis_dir,
        max_samples=4,  # Number of samples to save per epoch
        every_n_epochs=args.check_val_every_n_epoch  # Save every epoch
    )
    # bs_finder = BatchSizeFinder(mode='binsearch', init_val=config.batch_size)
    callbacks = [lr_monitor, checkpoint_callback, pc_sampler]
    if args.use_swa:
        swa_callback = StochasticWeightAveraging(swa_lrs=1e-2)
        callbacks.append(swa_callback)
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        strategy=args.strategy,
        callbacks=callbacks,
        enable_model_summary=True,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,  # Run validation once per 5 epochs
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        overfit_batches=args.overfit_batches,
    )

    if args.use_lr_finder:
        lr_finder = trainer.tuner.lr_find(clip_diffuser, max_lr=args.max_lr, datamodule=datamodule, update_attr=False)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("lr_finder.png")
        clip_diffuser.hparams.lr = lr_finder.suggestion()
        print(f"Learning rate set to {clip_diffuser.hparams.lr}")

    # Train the model
    trainer.fit(
        model=clip_diffuser,
        datamodule=datamodule,
        # ckpt_path="path/to/checkpoint.ckpt"  # Uncomment to resume training
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", "-r",type=str, default="scratch")
    parser.add_argument("--experiment_name", "-e", type=str, default="lightning_logs")
    parser.add_argument("--use_clip_cond", action="store_true")
    parser.add_argument("--overfit_batches", type=float, default=0.0)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--use_lr_finder", action="store_true")
    parser.add_argument("--use_swa", action="store_true")
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--devices", type=int, default=-1)
    parser.add_argument("--strategy", type=str, default="ddp_find_unused_parameters_false")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=30)
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--accumulate_grad_batches", type=int, default=4)
    parser.add_argument("--gradient_clip_val", type=float, default=0.5)
    parser.add_argument("--limit_train_batches", type=float, default=None)
    parser.add_argument("--limit_val_batches", type=float, default=None)
    parser.add_argument("--limit_test_batches", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.run_name = args.run_name + "_diff"
    if args.use_clip_cond:
        args.run_name += "_clip"
    else:
        args.run_name += "_dino"

    if args.overfit_batches >= 1:
        args.overfit_batches = 1 # just 1 batch, not 100%

    if args.debug:
        args.overfit_batches = 1
        args.max_epochs = 10
        args.log_every_n_steps = 1
        args.check_val_every_n_epoch = 1
        args.run_name += "_debug"
        args.experiment_name = "debug"

    train(args)