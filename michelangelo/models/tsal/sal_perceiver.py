# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional
from einops import repeat
import math

from michelangelo.models.modules import checkpoint
from michelangelo.models.modules.embedder import FourierEmbedder
from michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from michelangelo.models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer
)

from .tsal_base import ShapeAsLatentModule
from torch.nn import functional as F
import copy

class CrossAttentionEncoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 fourier_embedder: FourierEmbedder,
                 point_feats: int,
                 width: int,
                 heads: int,
                 layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        self.fourier_embedder = fourier_embedder
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:

        """

        bs = pc.shape[0]

        data = self.fourier_embedder(pc)
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        data = self.input_proj(data)

        query = repeat(self.query, "m c -> b m c", b=bs)
        latents = self.cross_attn(query, data)
        latents = self.self_attn(latents)

        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]

        Returns:
            dict
        """

        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttentionDecoder(nn.Module):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 out_channels: int,
                 dropout: float = 0.1,
                 num_layers: int = 2,
                 fourier_embedder: FourierEmbedder,
                 width: int,
                 heads: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.fourier_embedder = fourier_embedder

        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width)
        cross_attn_block = nn.TransformerDecoderLayer(d_model=width, nhead=heads, batch_first=True, dropout=dropout)
        # cross_attn_block = ResidualCrossAttentionBlock(
        #     device=device,
        #     dtype=dtype,
        #     width=width,
        #     heads=heads,
        #     init_scale=init_scale,
        #     qkv_bias=qkv_bias,
        #     flash=flash,
        # )
        self.cross_attn_decoder = nn.ModuleList([copy.deepcopy(cross_attn_block) for _ in range(num_layers)])
        offset_model = Mlp(width, hidden_features=width//4, out_features=3, drop=dropout)#nn.Linear(width, 3)
        self.offset_mlps = nn.ModuleList([copy.deepcopy(offset_model) for _ in range(num_layers)])
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)



    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        # complete_points = queries
        complete_points = queries.clone()
        queries = self.fourier_embedder(queries)
        queries = self.query_proj(queries)
        intermediate_completions = []
        for i, layer in enumerate(self.cross_attn_decoder):
            # cross attend with incomplete points
            queries = layer(queries, latents)
            # queries = self.ln_post(queries)
            offset = F.hardtanh(self.offset_mlps[i](queries))
            complete_points = (complete_points + offset) / 2.0

            intermediate_completions.append(complete_points)
        
        # queries = torch.cat([queries, incomplete_points], dim=-1)
        # complete_points = self.offset_model(queries)
        # complete_points = complete_points.clamp(-1.0, 1.0)
        complete_points = torch.cat(intermediate_completions, dim=1)
        # normalize between -1 and 1
        return complete_points

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 dropout: float,
                 num_decoder_layers_cross_attn: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.num_latents = num_latents
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.embed_dim = embed_dim
        if embed_dim > 0:
            # VAE embed
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

        # geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=3, # for point cloud reconstruction
            dropout=dropout,
            num_layers=num_decoder_layers_cross_attn,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            sample_posterior (bool):

        Returns:
            latents (torch.FloatTensor)
            center_pos (torch.FloatTensor or None):
            posterior (DiagonalGaussianDistribution or None):
        """

        latents, center_pos = self.encoder(pc, feats)

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                latents = posterior.sample()
            else:
                latents = posterior.mode()

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor):
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        logits = self.geo_decoder(queries, latents)#.squeeze(-1)
        return logits

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            logits (torch.FloatTensor): [B, P]
            center_pos (torch.FloatTensor): [B, M, 3]
            posterior (DiagonalGaussianDistribution or None).

        """

        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(latents)
        logits = self.query_geometry(volume_queries, latents)

        return logits, center_pos, posterior


class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):

    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 num_latents: int,
                 point_feats: int = 0,
                 embed_dim: int = 0,
                 num_freqs: int = 8,
                 dropout: float = 0.0,
                 num_decoder_layers_cross_attn: int = 2,
                 include_pi: bool = True,
                 width: int,
                 heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 init_scale: float = 0.25,
                 qkv_bias: bool = True,
                 flash: bool = False,
                 use_ln_post: bool = False,
                 use_checkpoint: bool = False):

        super().__init__(
            device=device,
            dtype=dtype,
            dropout=dropout,
            num_decoder_layers_cross_attn=num_decoder_layers_cross_attn,
            num_latents=1 + num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.width = width

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, c]
            sample_posterior (bool):

        Returns:
            shape_embed (torch.FloatTensor)
            kl_embed (torch.FloatTensor):
            posterior (DiagonalGaussianDistribution or None):
        """

        shape_embed, latents = self.encode_latents(pc, feats)
        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)

        return shape_embed, kl_embed, posterior

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):

        x, _ = self.encoder(pc, feats)

        shape_embed = x[:, 0]
        latents = x[:, 1:]

        return shape_embed, latents

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents

        return kl_embed, posterior

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """

        Args:
            pc (torch.FloatTensor): [B, N, 3]
            feats (torch.FloatTensor or None): [B, N, C]
            volume_queries (torch.FloatTensor): [B, P, 3]
            sample_posterior (bool):

        Returns:
            shape_embed (torch.FloatTensor): [B, projection_dim]
            logits (torch.FloatTensor): [B, M]
            posterior (DiagonalGaussianDistribution or None).

        """

        shape_embed, kl_embed, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(kl_embed)
        logits = self.query_geometry(volume_queries, latents)

        return shape_embed, logits, posterior