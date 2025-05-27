import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from .sita_vae import SITAVAE
from .diffusion import ConditionalASLUDTDenoiser, timestep_embedding


class EndToEndModel(nn.Module):
    """
    End-to-end model combining SITA-VAE and diffusion in latent space
    """
    def __init__(
        self,
        # SITA-VAE parameters
        num_latents: int = 256,
        embed_dim: int = 64,
        point_feats: int = 3,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        
        # Diffusion parameters
        diffusion_width: int = 768,
        diffusion_layers: int = 6,
        diffusion_heads: int = 12,
        
        # Training parameters
        diffusion_steps: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device
        self.diffusion_steps = diffusion_steps
        
        # Initialize SITA-VAE
        self.sita_vae = SITAVAE(
            num_latents=num_latents,
            embed_dim=embed_dim,
            point_feats=point_feats,
            clip_model_name=clip_model_name,
            device=device
        )
        
        # Initialize diffusion model
        self.diffusion = ConditionalASLUDTDenoiser(
            input_channels=embed_dim,
            output_channels=embed_dim,
            n_ctx=num_latents,
            width=diffusion_width,
            layers=diffusion_layers,
            heads=diffusion_heads,
            context_dim=embed_dim * 2,  # CLIP embedding dim
            use_checkpoint=False
        )
        
        # Loss weights
        self.recon_weight = 1.0
        self.kl_weight = 0.001
        self.contrast_weight = 0.1
        self.diffusion_weight = 1.0
        
        # For sampling
        self.betas = torch.linspace(1e-4, 0.02, diffusion_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion process: q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for the diffusion model
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Sample noisy latents
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        model_out = self.diffusion(x_t, t, context=condition)
        
        # Simple MSE loss
        loss = F.mse_loss(model_out, noise)
        
        return {"loss": loss, "pred_noise": model_out}
    
    def forward(
        self,
        point_cloud: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        texts: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> Dict[str, Any]:
        """
        Forward pass for training or inference
        
        Args:
            point_cloud: [B, N, 3+C] Input point cloud
            images: Optional [B, 3, H, W] images for conditioning
            texts: Optional list of text strings for conditioning
            mode: 'train' or 'sample'
            
        Returns:
            Dict containing losses and outputs
        """
        batch_size = point_cloud.size(0)
        
        # Encode to get clean latents
        with torch.no_grad():
            clean_latents, _, _ = self.sita_vae.encode(point_cloud)
            
            # Get condition (CLIP embeddings)
            if images is not None or texts is not None:
                # In practice, get CLIP embeddings here
                condition = torch.randn(batch_size, self.sita_vae.encoder.embed_dim * 2, 
                                      device=self.device)
            else:
                condition = None
        
        if mode == "train":
            # Sample random timesteps
            t = torch.randint(0, self.diffusion_steps, (batch_size,), 
                            device=self.device).long()
            
            # Compute diffusion loss
            diffusion_losses = self.p_losses(
                x_start=clean_latents,
                t=t,
                condition=condition,
                noise=None
            )
            
            # Total loss
            total_loss = self.diffusion_weight * diffusion_losses["loss"]
            
            return {
                "loss": total_loss,
                "diffusion_loss": diffusion_losses["loss"],
                "latents": clean_latents.detach()
            }
            
        elif mode == "sample":
            # Sample from the diffusion model
            return self.sample(condition=condition, shape=clean_latents.shape)
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @torch.no_grad()
    def sample(
        self,
        condition: Optional[torch.Tensor] = None,
        shape: Optional[Tuple[int, int, int]] = None,
        num_samples: int = 1,
        guidance_scale: float = 3.0
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the diffusion model
        """
        if shape is None:
            shape = (num_samples, self.sita_vae.num_latents, self.sita_vae.embed_dim)
            
        batch_size = shape[0]
        device = next(self.parameters()).device
        
        # Start with random noise
        x_t = torch.randn(shape, device=device)
        
        # Sample loop
        for t in reversed(range(self.diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            if guidance_scale > 1.0 and condition is not None:
                # Classifier-free guidance
                # Unconditional prediction
                model_out_uncond = self.diffusion(x_t, t_tensor, context=None)
                
                # Conditional prediction
                model_out_cond = self.diffusion(x_t, t_tensor, context=condition)
                
                # Combine with guidance scale
                model_out = model_out_uncond + guidance_scale * (model_out_cond - model_out_uncond)
            else:
                model_out = self.diffusion(x_t, t_tensor, context=condition)
            
            # Update x_t
            if t > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
                
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_t_prev = self.alphas_cumprod[t-1] if t > 0 else 1.0
            
            beta_t = self.betas[t]
            
            # DDIM sampling step
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * model_out) / torch.sqrt(alpha_bar_t)
            
            # Clamp for stability
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
            
            # Update x_t
            if t > 0:
                x_t = (
                    torch.sqrt(alpha_bar_t_prev) * pred_x0 +
                    torch.sqrt(1 - alpha_bar_t_prev) * noise
                )
        
        # Decode the final latents
        with torch.no_grad():
            decoded = self.sita_vae.decode(x_t)
        
        return {
            "latents": x_t,
            "decoded": decoded
        }


def extract(a, t, x_shape):
    """
    Extract elements from a tensor 'a' using indices 't' and reshape to match 'x_shape'.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
