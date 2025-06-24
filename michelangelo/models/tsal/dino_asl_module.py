# -*- coding: utf-8 -*-

import torch
from torch import nn
from michelangelo.models.tsal.tsal_base import AlignedShapeAsLatentModule
from michelangelo.models.conditional_encoders.encoder_factory import DinoImageEmbedder


class DinoAlignedShapeAsLatentModule(AlignedShapeAsLatentModule):

    def __init__(self, *,
                 shape_model,
                 clip_model_version: str = "facebook/dinov2-large",
                 use_contrastive: bool = True):

        super().__init__()

        self.use_contrastive = use_contrastive
        if self.use_contrastive:
            self.clip_model: DinoImageEmbedder = DinoImageEmbedder(version=clip_model_version)
            for params in self.clip_model.parameters():
                params.requires_grad = False
        else:
            self.clip_model = None

        self.shape_model = shape_model
        self.shape_projection = nn.Parameter(torch.empty(self.shape_model.width, self.clip_model.model.config.vision_config.hidden_size))
        nn.init.normal_(self.shape_projection, std=self.clip_model.model.config.vision_config.hidden_size ** -0.5)

    def set_shape_model_only(self):
        self.clip_model = None

    def encode_shape_embed(self, surface, return_latents: bool = False):
        """

        Args:
            surface (torch.FloatTensor): [bs, n, 3 + c]
            return_latents (bool):

        Returns:
            x (torch.FloatTensor): [bs, projection_dim]
            shape_latents (torch.FloatTensor): [bs, m, d]
        """

        pc = surface[..., 0:3]
        feats = surface[..., 3:]

        shape_embed, shape_latents = self.shape_model.encode_latents(pc, feats)
        x = shape_embed @ self.shape_projection

        if return_latents:
            return x, shape_latents
        else:
            return x

    def encode_image_embed(self, image):
        """

        Args:
            image (torch.FloatTensor): [bs, 3, h, w]

        Returns:
            x (torch.FloatTensor): [bs, projection_dim]
        """

        x = self.clip_model.encode(image).mean(dim=1)

        return x

    # def encode_text_embed(self, text):
    #     x = self.clip_model.encode(text)
    #     return x

    def forward(self, surface, image, text):
        """

        Args:
            surface (torch.FloatTensor):
            image (torch.FloatTensor): [bs, 3, 224, 224]
            text (torch.LongTensor): [bs, num_templates, 77]

        Returns:
            embed_outputs (dict): the embedding outputs, and it contains:
                - image_embed (torch.FloatTensor):
                - text_embed (torch.FloatTensor):
                - shape_embed (torch.FloatTensor):
                - logit_scale (float):
        """

        # # text embedding
        # text_embed_all = []
        # for i in range(text.shape[0]):
        #     text_for_one_sample = text[i]
        #     text_embed = self.encode_text_embed(text_for_one_sample)
        #     text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        #     text_embed = text_embed.mean(dim=0)
        #     text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        #     text_embed_all.append(text_embed)
        # text_embed_all = torch.stack(text_embed_all)

        # b = text.shape[0]
        # text_tokens = rearrange(text, "b t l -> (b t) l")
        # text_embed = self.encode_text_embed(text_tokens)
        # text_embed = rearrange(text_embed, "(b t) d -> b t d", b=b)
        # text_embed = text_embed.mean(dim=1)
        # text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        # image embedding

        # shape embedding
        shape_embed, shape_latents = self.encode_shape_embed(surface, return_latents=True)
        if self.use_contrastive:
            image_embed = self.encode_image_embed(image)
        else:
            image_embed = torch.zeros_like(shape_embed)

        embed_outputs = {
            "image_embed": image_embed,
            "text_embed": None,
            "shape_embed": shape_embed,
            "logit_scale": 1.0 #self.clip_model.logit_scale#.exp()
        }

        return embed_outputs, shape_latents
