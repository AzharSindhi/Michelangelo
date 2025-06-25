# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from typing import Union
from functools import partial

from michelangelo.utils import instantiate_from_config
from michelangelo.utils.misc import get_obj_from_str

from .inference_utils import extract_geometry
from .tsal_base import (
    AlignedShapeAsLatentModule,
    ShapeAsLatentModule,
    Latent2MeshOutput,
    AlignedMeshOutput
)
import numpy as np
from einops import repeat
from torch import nn
from michelangelo.models.pointnet_vae.pointnet2_vae import PointNet2CloudCondition
from michelangelo.models.tsal.loss import ContrastKLNearFar

class AlignedShapeAsLatentPLModule(pl.LightningModule):

    def __init__(self, *,
                 shape_module_cfg,
                 aligned_module_cfg,
                 loss_cfg,
                 optimizer_cfg: Optional[DictConfig] = None,
                 ckpt_path: Optional[str] = None,
                 ignore_keys: Union[Tuple[str], List[str]] = (),
                 dtype=torch.float32,
                 device="cuda",
                 numpoints: int = 4096):

        super().__init__()
        
        cls = get_obj_from_str(shape_module_cfg.target)
        shape_model: PointNet2CloudCondition = cls(shape_module_cfg.params)
        self.model: AlignedShapeAsLatentModule = instantiate_from_config(
            aligned_module_cfg, shape_model=shape_model
        )

        self.loss: ContrastKLNearFar = instantiate_from_config(loss_cfg)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            print("No checkpoint provided, learning from scratch")
        
        if optimizer_cfg is not None:
            self.optimizer_cfg = optimizer_cfg
            self.learning_rate = optimizer_cfg.optimizer.params.lr
        else:
            self.learning_rate = 1.e-4
        
        self.numpoints = numpoints
        # for callbacks and logging
        self.last_train_output = None
        self.last_val_output = None
        self.save_hyperparameters()

    def set_shape_model_only(self):
        self.model.set_shape_model_only()

    @property
    def latent_shape(self):
        return self.model.shape_model.latent_shape

    @property
    def zero_rank(self):
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

    def init_from_ckpt(self, path, ignore_keys=()):
        state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

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
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self):
        # lr = self.learning_rate

        trainable_parameters = list(self.model.parameters())

        if self.optimizer_cfg is None:
            optimizer = torch.optim.AdamW(trainable_parameters, lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=1e-3)
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=trainable_parameters)

        scheduler = None
        if hasattr(self, 'optimizer_cfg') and hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                optimizer=optimizer,
            )
            scheduler = {
                "scheduler": scheduler_func,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "train/total_loss"
            }

        return [optimizer], [scheduler]

    def forward(self,
                surface: torch.FloatTensor,
                image: torch.FloatTensor,
                text: torch.FloatTensor,
                incomplete_points: torch.FloatTensor):

        """

        Args:
            surface (torch.FloatTensor):
            incomplete_points (torch.FloatTensor):
            image (torch.FloatTensor):
            text (torch.FloatTensor):

        Returns:
            embed_outputs (dict):
            recon_pc (torch.FloatTensor):
            posterior (DiagonalGaussianDistribution):

        """

        posterior = None
        embed_outputs, complete_shape_zq, partial_shape_zq = self.model(surface, incomplete_points, image)
        # shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_zq)
        recon_pc, recon_pc_partial = self.model.shape_model.decode(complete_shape_zq, partial_shape_zq, incomplete_points)
        
        return embed_outputs, recon_pc, recon_pc_partial, posterior

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_zq, posterior = self.model.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_zq

    def decode(self,
               z_q,
               incomplete_points: torch.FloatTensor,
            #    bounds: Union[Tuple[float], List[float], float] = 1.1,
            #    octree_depth: int = 7,
            #    num_chunks: int = 10000
               ):
        
        # generate point cloud from latent (decoder)

        latents = self.model.shape_model.decode(z_q)  # latents: [bs, num_latents, dim]
        recon_pc = self.model.shape_model.query_geometry(incomplete_points, latents)
        return recon_pc

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int,
                      optimizer_idx: int=0) -> torch.FloatTensor:
        """

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor): [bs, n_surface, (3 + input_dim)]
                - image (torch.FloatTensor): [bs, 3, 224, 224]
                - text (torch.FloatTensor): [bs, num_templates, 77]
                - geo_points (torch.FloatTensor): [bs, n_pts, (3 + 1)]

            batch_idx (int):

            optimizer_idx (int):

        Returns:
            loss (torch.FloatTensor):

        """

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        incomplete_points = batch["incomplete_points"][..., 0:3]
        # shape_labels = torch.zeros_like(batch["incomplete_points"][..., -1]) #batch["geo_points"][..., -1]

        embed_outputs, reconstructed_pc, reconstructed_pc_partial, posteriors = self(surface, image, text, incomplete_points)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            reconstructed_pc=reconstructed_pc,
            reconstructed_pc_partial=reconstructed_pc_partial,
            gt_pc=surface[..., :3],
            gt_partial=incomplete_points,
            split="train"
        )

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=reconstructed_pc.shape[0],
                      sync_dist=True, on_step=False, on_epoch=True)

        self.last_train_output_pc = reconstructed_pc.detach().clone().cpu()
        self.last_train_output_pc_partial = reconstructed_pc_partial.detach().clone().cpu()

        return aeloss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        incomplete_points = batch["incomplete_points"][..., 0:3]
        # shape_labels = torch.zeros_like(batch["incomplete_points"][..., -1])#batch["geo_points"][..., -1]

        embed_outputs, reconstructed_pc, reconstructed_pc_partial, posteriors = self(surface, image, text, incomplete_points)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            reconstructed_pc=reconstructed_pc,
            reconstructed_pc_partial=reconstructed_pc_partial,
            gt_pc=surface[..., :3],
            gt_partial=incomplete_points,
            split="val"
        )
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=reconstructed_pc.shape[0],
                      sync_dist=True, on_step=False, on_epoch=True)

        self.last_val_output_pc = reconstructed_pc.detach().clone().cpu()
        self.last_val_output_pc_partial = reconstructed_pc_partial.detach().clone().cpu()

        return aeloss
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        incomplete_points = batch["incomplete_points"][..., 0:3]
        # shape_labels = torch.zeros_like(batch["incomplete_points"][..., -1])#batch["geo_points"][..., -1]

        embed_outputs, reconstructed_pc, reconstructed_pc_partial, posteriors = self(surface, image, text, incomplete_points)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            reconstructed_pc=reconstructed_pc,
            reconstructed_pc_partial=reconstructed_pc_partial,
            gt_pc=surface[..., :3],
            gt_partial=incomplete_points,
            split="val"
        )
        # self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=reconstructed_pc.shape[0],
        #               sync_dist=True, on_step=False, on_epoch=True)

        self.last_predict_output_pc = reconstructed_pc.detach().clone().cpu()
        self.last_predict_output_pc_partial = reconstructed_pc_partial.detach().clone().cpu()

        return aeloss
    
    def visual_alignment(self,
                         surface: torch.FloatTensor,
                         image: torch.FloatTensor,
                         text: torch.FloatTensor,
                         description: Optional[List[str]] = None,
                         bounds: Union[Tuple[float], List[float]] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                         octree_depth: int = 7,
                         num_chunks: int = 10000) -> List[AlignedMeshOutput]:

        """

        Args:
            surface:
            image:
            text:
            description:
            bounds:
            octree_depth:
            num_chunks:

        Returns:
            mesh_outputs (List[AlignedMeshOutput]): the mesh outputs list.

        """

        outputs = []

        device = surface.device
        bs = surface.shape[0]

        embed_outputs, shape_z = self.model(surface, image, text)

        # calculate the similarity
        image_embed = embed_outputs["image_embed"]
        text_embed = embed_outputs["text_embed"]
        shape_embed = embed_outputs["shape_embed"]

        # normalized features
        shape_embed = F.normalize(shape_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # B x B
        shape_text_similarity = (100.0 * shape_embed @ text_embed.T).softmax(dim=-1)

        # B x B
        shape_image_similarity = (100.0 * shape_embed @ image_embed.T).softmax(dim=-1)

        # shape reconstruction
        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

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

            out = AlignedMeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            out.surface = surface[i].cpu().numpy()
            out.image = image[i].cpu().numpy()
            if description is not None:
                out.text = description[i]
            out.shape_text_similarity = shape_text_similarity[i, i]
            out.shape_image_similarity = shape_image_similarity[i, i]

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

        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

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
