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

        shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, dtype=dtype, device=device
        )
        self.model: AlignedShapeAsLatentModule = instantiate_from_config(
            aligned_module_cfg, shape_model=shape_model
        )

        self.loss = instantiate_from_config(loss_cfg)
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

    def configure_optimizers(self) -> Tuple[List, List]:
        # lr = self.learning_rate

        trainable_parameters = list(self.model.parameters())

        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(trainable_parameters, lr=self.learning_rate, betas=(0.9, 0.99), weight_decay=1e-3)]
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=trainable_parameters)
            optimizers = [optimizer]

        schedulers = []
        if hasattr(self, 'optimizer_cfg') and hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                optimizer=optimizers[0],
            )
            scheduler = {
                "scheduler": scheduler_func,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/total_loss"
            }
            schedulers = [scheduler]

        return optimizers, schedulers

    def forward(self,
                surface: torch.FloatTensor,
                image: torch.FloatTensor,
                text: torch.FloatTensor,
                volume_queries: torch.FloatTensor):

        """

        Args:
            surface (torch.FloatTensor):
            image (torch.FloatTensor):
            text (torch.FloatTensor):
            volume_queries (torch.FloatTensor):

        Returns:

        """

        embed_outputs, shape_z = self.model(surface, image, text)
        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        
        # add position embedding
        # volume_queries_learnable = volume_queries_learnable + self.position_transform(volume_queries)
        # incomplete_points = volume_queries.clone()
        recon_pc = self.model.shape_model.query_geometry(volume_queries, latents)
        # recon_pc = recon_pc + incomplete_points
        # recon_pc = recon_pc.clamp(-1.0, 1.0)
        return embed_outputs, recon_pc, posterior

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):

        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_zq, posterior = self.model.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_zq

    def decode(self,
               z_q,
               volume_queries: torch.FloatTensor,
            #    bounds: Union[Tuple[float], List[float], float] = 1.1,
            #    octree_depth: int = 7,
            #    num_chunks: int = 10000
               ):
        
        # generate point cloud from latent (decoder)

        latents = self.model.shape_model.decode(z_q)  # latents: [bs, num_latents, dim]
        recon_pc = self.model.shape_model.query_geometry(volume_queries, latents)
        return recon_pc

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
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

        volume_queries = batch["incomplete_points"][..., 0:3] #batch["geo_points"][..., 0:3]
        # shape_labels = torch.zeros_like(batch["incomplete_points"][..., -1]) #batch["geo_points"][..., -1]

        embed_outputs, reconstructed_pc, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            reconstructed_pc=reconstructed_pc,
            gt_pc=surface[..., :3],
            split="train"
        )

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=reconstructed_pc.shape[0],
                      sync_dist=True, on_step=False, on_epoch=True)

        self.last_train_output = reconstructed_pc.detach().clone().cpu()
        self.last_train_output = self.last_train_output[:, -self.numpoints:]

        return aeloss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["incomplete_points"][..., 0:3]  #batch["geo_points"][..., 0:3]
        # shape_labels = torch.zeros_like(batch["incomplete_points"][..., -1])#batch["geo_points"][..., -1]

        embed_outputs, reconstructed_pc, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            reconstructed_pc=reconstructed_pc,
            gt_pc=surface[..., :3],
            split="val"
        )
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=reconstructed_pc.shape[0],
                      sync_dist=True, on_step=False, on_epoch=True)

        self.last_val_output = reconstructed_pc.detach().clone().cpu()
        self.last_val_output = self.last_val_output[:, -self.numpoints:]

        return aeloss
    
    def predict_step(self, batch, batch_idx, dataloader_idx = 0):
        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["incomplete_points"][..., 0:3]  #batch["geo_points"][..., 0:3]
        # shape_labels = torch.zeros_like(batch["incomplete_points"][..., -1])#batch["geo_points"][..., -1]

        embed_outputs, reconstructed_pc, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            reconstructed_pc=reconstructed_pc,
            gt_pc=surface[..., :3],
            split="val"
        )
        # self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=reconstructed_pc.shape[0],
        #               sync_dist=True, on_step=False, on_epoch=True)

        self.last_predict_output = reconstructed_pc.detach().clone().cpu()
        self.last_predict_output = self.last_predict_output[:, -self.numpoints:]

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
