# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional
from omegaconf import DictConfig

import torch
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from typing import Union
from functools import partial

from michelangelo.utils import instantiate_from_config

from .inference_utils import extract_geometry
from .tsal_base import (
    ShapeAsLatentModule,
    Latent2MeshOutput,
    Point2MeshOutput
)


class ShapeAsLatentPLModule(pl.LightningModule):

    def __init__(self, *,
                 module_cfg,
                 loss_cfg,
                 optimizer_cfg: Optional[DictConfig] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = ()):

        super().__init__()

        self.sal: ShapeAsLatentModule = instantiate_from_config(module_cfg, device=None, dtype=None)

        self.loss = instantiate_from_config(loss_cfg)

        self.optimizer_cfg = optimizer_cfg

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters()

    @property
    def latent_shape(self):
        return self.sal.latent_shape

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        # optimizers = [torch.optim.AdamW(self.sal.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)]
        # optimizers = [torch.optim.AdamW(self.sal.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]

        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(self.sal.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]
            schedulers = []
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=self.sal.parameters())
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            optimizers = [optimizer]
            schedulers = [scheduler]

        return optimizers, schedulers

    def forward(self,
                pc: torch.FloatTensor,
                feats: Optional[torch.FloatTensor] = None,
                sample_posterior: bool = True):
        """
        Forward pass for point cloud generation.

        Args:
            pc: Input point cloud [B, N, 3]
            feats: Optional features [B, N, C]
            sample_posterior: Whether to sample from posterior

        Returns:
            points: Generated point cloud [B, num_points, 3]
            center_pos: Center positions [B, M, 3]
            posterior: Posterior distribution or None
        """
        points, center_pos, posterior = self.sal(
            pc=pc, 
            feats=feats, 
            sample_posterior=sample_posterior
        )
        return points, center_pos, posterior

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        latents, center_pos, posterior = self.sal.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return latents

    def decode(self,
               z_q,
               bounds: Union[Tuple[float], List[float], float] = 1.1,
               octree_depth: int = 7,
               num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        latents = self.sal.decode(z_q)  # latents: [bs, num_latents, dim]
        outputs = self.latent2mesh(latents, bounds=bounds, octree_depth=octree_depth, num_chunks=num_chunks)

        return outputs

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """
        Training step for point cloud reconstruction.

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor): [bs, n_points, (3 + input_dim)]
            batch_idx (int): batch index
            optimizer_idx (int): optimizer index

        Returns:
            loss (torch.FloatTensor): computed loss
        """
        # Input point cloud and ground truth points
        pc = batch["surface"][..., 0:3]  # [B, N, 3]
        feats = batch["surface"][..., 3:] if batch["surface"].shape[-1] > 3 else None
        
        # Use input points as ground truth
        gt_points = pc
        
        # Forward pass
        pred_points, _, posterior = self.sal(pc=pc, feats=feats)
        
        # Compute loss
        loss, log_dict = self.loss(
            posteriors=posterior,
            pred_points=pred_points,
            gt_points=gt_points,
            split="train"
        )
        
        # Log metrics
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            batch_size=pred_points.shape[0],
            sync_dist=True,
        )
        
        return loss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:
        """
        Validation step for point cloud reconstruction.

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor): [bs, n_points, (3 + input_dim)]
            batch_idx (int): batch index

        Returns:
            loss (torch.FloatTensor): computed loss
        """
        # Input point cloud and ground truth points
        pc = batch["surface"][..., 0:3]  # [B, N, 3]
        feats = batch["surface"][..., 3:] if batch["surface"].shape[-1] > 3 else None
        
        # Use input points as ground truth
        gt_points = pc
        
        # Forward pass (don't sample from posterior during validation)
        with torch.no_grad():
            pred_points, _, posterior = self.sal(pc=pc, feats=feats, sample_posterior=False)
            
            # Compute loss
            loss, log_dict = self.loss(
                posteriors=posterior,
                pred_points=pred_points,
                gt_points=gt_points,
                split="val"
            )
        
        # Log metrics
        self.log_dict(
            log_dict,
            prog_bar=True,
            logger=True,
            batch_size=pred_points.shape[0],
            sync_dist=True,
        )
        
        return loss

    def point2mesh(self,
                   pc: torch.FloatTensor,
                   feats: torch.FloatTensor,
                   bounds: Union[Tuple[float], List[float]] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                   octree_depth: int = 7,
                   num_chunks: int = 10000) -> List[Point2MeshOutput]:

        """

        Args:
            pc:
            feats:
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.

        """

        outputs = []

        device = pc.device
        bs = pc.shape[0]

        # 1. point encoder + latents transformer
        latents, center_pos, posterior = self.sal.encode(pc, feats)
        latents = self.sal.decode(latents)  # latents: [bs, num_latents, dim]

        geometric_func = partial(self.sal.query_geometry, latents=latents)

        # 2. decode geometry
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=bs,
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Point2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            out.pc = torch.cat([pc[i], feats[i]], dim=-1).cpu().numpy()

            if center_pos is not None:
                out.center = center_pos[i].cpu().numpy()

            outputs.append(out)

        return outputs

    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:

        """

        Args:
            latents: [bs, num_latents, dim]
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[MeshOutput]): the mesh outputs list.

        """

        outputs = []

        geometric_func = partial(self.sal.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs
