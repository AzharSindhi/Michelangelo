import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math
from einops import rearrange, repeat

class AlignedShapeLatentPerceiver(nn.Module):
    """
    Aligned Shape Latent Perceiver from the original Michelangelo implementation
    Matches the configuration in shapevae-256.yaml
    """
    def __init__(
        self,
        num_latents: int = 256,
        embed_dim: int = 64,
        point_feats: int = 3,  # XYZ coordinates + normals
        num_freqs: int = 8,
        include_pi: bool = False,
        heads: int = 12,
        width: int = 768,
        num_encoder_layers: int = 8,
        num_decoder_layers: int = 16,
        use_ln_post: bool = True,
        init_scale: float = 0.25,
        qkv_bias: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.width = width
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            in_dim=point_feats,
            num_freqs=num_freqs,
            include_pi=include_pi
        )
        
        # Input projection
        self.input_proj = nn.Linear(self.pos_encoder.out_dim, width)
        
        # Learnable query tokens
        self.query_embeddings = nn.Parameter(
            torch.randn(1, num_latents, width) * init_scale
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4*width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Transformer decoder for neural field
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=4*width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # Output projection
        self.output_proj = nn.Linear(width, embed_dim)
        
        # Layer norm
        self.ln_post = nn.LayerNorm(embed_dim) if use_ln_post else nn.Identity()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def encode(
        self,
        pc: torch.Tensor,
        feats: torch.Tensor,
        sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        Encode point cloud to latent space
        
        Args:
            pc: [B, N, 3] point coordinates
            feats: [B, N, C] point features (normals, etc.)
            sample_posterior: Whether to sample from the posterior
            
        Returns:
            latents: [B, num_latents, embed_dim] latent codes
            center_pos: [B, 3] center position (not used)
            posterior: Distribution object
        """
        # Concatenate coordinates and features
        x = torch.cat([pc, feats], dim=-1)  # [B, N, 3+C]
        
        # Positional encoding
        x = self.pos_encoder(x)  # [B, N, D]
        
        # Project to transformer dimension
        x = self.input_proj(x)  # [B, N, width]
        
        # Add learnable position embeddings
        batch_size = x.shape[0]
        pos_emb = self.query_embeddings.repeat(batch_size, 1, 1)  # [B, num_latents, width]
        
        # Pass through transformer encoder
        memory = self.encoder(x)  # [B, N, width]
        
        # Cross-attention to get latents
        latents = self.decoder(
            tgt=pos_emb,
            memory=memory
        )  # [B, num_latents, width]
        
        # Project to latent space
        latents = self.output_proj(latents)  # [B, num_latents, embed_dim]
        latents = self.ln_post(latents)
        
        # For VAE, predict mean and logvar
        mean = latents
        logvar = torch.zeros_like(mean)
        
        # Reparameterization trick
        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latents = mean + eps * std
        else:
            latents = mean
        
        # Create posterior distribution
        posterior = torch.distributions.Normal(mean, torch.exp(0.5 * logvar))
        
        return latents, torch.zeros(batch_size, 3, device=latents.device), posterior
    
    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to neural field
        
        Args:
            z_q: [B, num_latents, embed_dim] latent codes
            
        Returns:
            [B, 1, 3] center position (dummy for compatibility)
        """
        # In the full implementation, this would query points in 3D space
        # and return occupancy/SDF values
        return torch.zeros(z_q.size(0), 1, 3, device=z_q.device)


class CLIPAlignedShapeAsLatentModule(nn.Module):
    """
    CLIP Aligned Shape Latent Module from the original implementation
    """
    def __init__(
        self,
        clip_model_version: str = "./checkpoints/clip/clip-vit-large-patch14",
        embed_dim: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        
        # In the full implementation, we'd load CLIP here
        # self.clip_model = CLIPModel.from_pretrained(clip_model_version).to(device)
        # self.clip_model.requires_grad_(False)
        
        # Projection layers
        self.shape_proj = nn.Linear(embed_dim, embed_dim)
        self.image_proj = nn.Linear(1024, embed_dim)  # CLIP image dim
        self.text_proj = nn.Linear(768, embed_dim)     # CLIP text dim
        
        # Initialize projections
        nn.init.normal_(self.shape_proj.weight, std=0.02)
        nn.init.normal_(self.image_proj.weight, std=0.02)
        nn.init.normal_(self.text_proj.weight, std=0.02)
        
        # Loss weights
        self.contrast_weight = 0.1
        self.near_weight = 0.1
        self.kl_weight = 0.001
    
    def forward(
        self,
        shape_embeddings: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        texts: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for shape-image-text alignment
        
        Returns:
            Dict containing losses
        """
        losses = {}
        
        # Project shape embeddings
        shape_emb = self.shape_proj(shape_embeddings.mean(dim=1))  # [B, embed_dim]
        shape_emb = F.normalize(shape_emb, dim=-1)
        
        # Compute losses if images or texts are provided
        if images is not None:
            # In the full implementation, we'd use CLIP's image encoder
            # image_features = self.clip_model.get_image_features(images)
            # image_emb = self.image_proj(image_features)
            # image_emb = F.normalize(image_emb, dim=-1)
            # 
            # # Compute contrastive loss
            # logits = shape_emb @ image_emb.t() / 0.07  # Temperature
            # labels = torch.arange(len(logits), device=self.device)
            # loss_i = F.cross_entropy(logits, labels)
            # loss_t = F.cross_entropy(logits.t(), labels)
            # losses["image_contrast"] = (loss_i + loss_t) / 2
            pass
            
        if texts is not None:
            # Similar for text
            pass
            
        # Total loss
        losses["total"] = sum([
            self.contrast_weight * losses.get("image_contrast", 0.0),
            self.contrast_weight * losses.get("text_contrast", 0.0),
            self.near_weight * losses.get("near_loss", 0.0),
            self.kl_weight * losses.get("kl_loss", 0.0)
        ])
        
        return losses


class PositionalEncoding(nn.Module):
    """
    Positional encoding for point coordinates
    """
    def __init__(self, in_dim: int, num_freqs: int, include_pi: bool = False):
        super().__init__()
        self.num_freqs = num_freqs
        self.in_dim = in_dim
        self.include_pi = include_pi
        
        # Create frequency bands
        self.freq_bands = 2.0 ** torch.linspace(
            0.0, num_freqs - 1, num_freqs
        )
        
        # Output dimension is in_dim * (2 * num_freqs + include_pi)
        self.out_dim = in_dim * (2 * num_freqs + (1 if include_pi else 0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., in_dim] input coordinates
            
        Returns:
            [..., out_dim] positionally encoded features
        """
        # Expand to [..., num_freqs, in_dim]
        x = x.unsqueeze(-2)  # [..., 1, in_dim]
        x = x * self.freq_bands.view(1, -1, 1).to(x.device)  # [..., num_freqs, in_dim]
        
        # Compute sin and cos
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # [..., num_freqs, 2 * in_dim]
        
        # Add original coordinates if needed
        if self.include_pi:
            x = torch.cat([x, x[..., :1, :] * math.pi], dim=-2)
        
        # Flatten last two dimensions
        return x.reshape(*x.shape[:-2], -1)  # [..., out_dim]


class SITAVAE(nn.Module):
    """
    Shape-Image-Text-Aligned VAE implementation matching the original Michelangelo codebase.
    Combines shape encoding with CLIP alignment for multimodal learning.
    """
    def __init__(
        self,
        num_latents: int = 256,
        embed_dim: int = 64,
        point_feats: int = 3,  # XYZ coordinates
        clip_model_version: str = "openai/clip-vit-large-patch14",
        use_checkpoint: bool = False,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.num_latents = num_latents
        self.embed_dim = embed_dim
        self.width = 768  # Hidden dimension for transformer
        
        # Initialize shape encoder/decoder
        self.shape_encoder = AlignedShapeLatentPerceiver(
            num_latents=num_latents,
            embed_dim=embed_dim,
            point_feats=point_feats,
            num_freqs=8,
            include_pi=False,
            heads=12,
            width=self.width,
            num_encoder_layers=8,
            num_decoder_layers=16,
            use_ln_post=True,
            init_scale=0.25,
            qkv_bias=False,
            use_checkpoint=use_checkpoint,
            device=device
        )
        
        # Initialize CLIP alignment module
        self.clip_aligner = CLIPAlignedShapeAsLatentModule(
            clip_model_version=clip_model_version,
            embed_dim=embed_dim,
            device=device
        )
        
        # Projection layers for latent space
        self.latent_proj = nn.Linear(self.width, embed_dim)
        self.latent_proj_inv = nn.Linear(embed_dim, self.width)
        
        # Initialize projections with CLIP's initialization scheme
        nn.init.normal_(self.latent_proj.weight, std=self.width ** -0.5)
        nn.init.normal_(self.latent_proj_inv.weight, std=self.width ** -0.5)
        
        # Loss weights from config
        self.kl_weight = 0.001
        self.contrast_weight = 0.1
        self.near_weight = 0.1
        
        # Logit scale for CLIP-style contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode(
        self,
        point_cloud: torch.Tensor,
        sample_posterior: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        Encode point cloud to latent space.
        
        Args:
            point_cloud: [B, N, 3+C] tensor of point clouds (XYZ + features)
            sample_posterior: Whether to sample from the posterior
            
        Returns:
            latents: [B, num_latents, embed_dim] latent codes
            center_pos: [B, 3] center position (dummy for compatibility)
            posterior: Distribution object
        """
        # Split point cloud into coordinates and features
        coords = point_cloud[..., :3].to(self.device)  # [B, N, 3]
        feats = point_cloud[..., 3:].to(self.device) if point_cloud.size(-1) > 3 else None  # [B, N, C]
        
        # Get shape embeddings and latents
        shape_embed, shape_latents = self.shape_encoder.encode_latents(coords, feats)
        
        # Project to latent space
        latents = self.latent_proj(shape_embed)
        
        # Get posterior distribution
        posterior = self.shape_encoder.encode_kl_embed(shape_latents, sample_posterior)
        
        # Sample from posterior if needed
        if sample_posterior:
            latents = posterior.sample()
        
        # Return dummy center position for compatibility
        center_pos = torch.zeros((latents.size(0), 3), device=self.device)
        
        return latents, center_pos, posterior
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes to neural field.
        
        Args:
            latents: [B, num_latents, embed_dim] latent codes
            
        Returns:
            [B, 1, 3] center position (dummy for compatibility)
        """
        # Project back to original dimension
        latents = self.latent_proj_inv(latents)
        
        # Decode to neural field
        return self.shape_encoder.decode(latents)
    
    def forward(
        self,
        point_cloud: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        texts: Optional[Union[List[str], torch.Tensor]] = None,
        sample_posterior: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional image/text conditioning.
        
        Args:
            point_cloud: [B, N, 3+C] tensor of point clouds
            images: Optional [B, 3, H, W] tensor of images
            texts: Optional list of text strings or pre-computed text embeddings
            sample_posterior: Whether to sample from the posterior
            
        Returns:
            Dict containing losses and outputs
        """
        # Encode point cloud
        coords = point_cloud[..., :3].to(self.device)
        feats = point_cloud[..., 3:].to(self.device) if point_cloud.size(-1) > 3 else None
        
        # Get shape embeddings and latents
        shape_embed, shape_latents = self.shape_encoder.encode_latents(coords, feats)
        
        # Get posterior distribution and sample if needed
        posterior = self.shape_encoder.encode_kl_embed(shape_latents, sample_posterior)
        if sample_posterior:
            shape_embed = posterior.sample()
        
        # Project to latent space
        latents = self.latent_proj(shape_embed)
        
        # Initialize outputs
        outputs = {
            'latents': latents,
            'posterior': posterior,
            'kl_loss': posterior.kl().mean() * self.kl_weight
        }
        
        # Compute CLIP alignment if conditioning is provided
        if images is not None or texts is not None:
            # Get CLIP embeddings
            clip_outputs = self.clip_aligner(
                surface=torch.cat([coords, feats], dim=-1) if feats is not None else coords,
                image=images,
                text=texts
            )
            
            # Add CLIP losses
            outputs.update({
                'contrast_loss': clip_outputs.get('contrast_loss', 0.0) * self.contrast_weight,
                'near_loss': clip_outputs.get('near_loss', 0.0) * self.near_weight,
                'logit_scale': clip_outputs.get('logit_scale', 1.0)
            })
            
            # Add total loss
            outputs['loss'] = outputs['kl_loss'] + outputs['contrast_loss'] + outputs['near_loss']
        else:
            outputs['loss'] = outputs['kl_loss']
        
        return outputs
    
    def get_latents(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Get latents without computing gradients for inference"""
        with torch.no_grad():
            latents, _, _ = self.encode(point_cloud, sample_posterior=False)
        return latents
    
    @torch.no_grad()
    def sample_prior(self, batch_size: int = 1) -> torch.Tensor:
        """Sample from the prior distribution"""
        device = next(self.parameters()).device
        shape = (batch_size, self.num_latents, self.embed_dim)
        return torch.randn(shape, device=device)
    
    def reconstruct(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """Reconstruct point cloud from latents"""
        with torch.no_grad():
            latents = self.get_latents(point_cloud)
            return self.decode(latents)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for shape-image-text alignment
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(
        self,
        shape_emb: torch.Tensor,
        modality_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between shape and modality embeddings
        
        Args:
            shape_emb: [B, D] shape embeddings
            modality_emb: [B, D] modality (image/text) embeddings
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        shape_emb = F.normalize(shape_emb, dim=-1)
        modality_emb = F.normalize(modality_emb, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(shape_emb, modality_emb.t()) / self.temperature
        
        # Labels are the diagonal elements
        batch_size = shape_emb.size(0)
        labels = torch.arange(batch_size, device=shape_emb.device)
        
        # Compute cross-entropy loss
        loss_i = self.cross_entropy(logits, labels)
        loss_t = self.cross_entropy(logits.t(), labels)
        
        return (loss_i + loss_t) / 2.0
