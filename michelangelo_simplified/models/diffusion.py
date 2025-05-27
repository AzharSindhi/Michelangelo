import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any, List, Union
from einops import rearrange, repeat

from .sita_vae import CLIPAlignedShapeAsLatentModule

class ConditionalASLUDTDenoiser(nn.Module):
    """
    Conditional Aligned Shape Latent Diffusion Model (ASLDM) denoiser
    Matches the configuration in image-ASLDM-256.yaml
    """
    def __init__(
        self,
        input_channels: int = 64,
        output_channels: int = 64,
        n_ctx: int = 256,
        width: int = 768,
        layers: int = 6,  # 2 * 6 + 1 = 13 layers total
        heads: int = 12,
        context_dim: int = 1024,  # CLIP text/image embedding dim
        init_scale: float = 1.0,
        skip_ln: bool = True,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.heads = heads
        self.context_dim = context_dim
        self.use_checkpoint = use_checkpoint
        
        # Time embedding
        time_embed_dim = width * 4
        self.time_embed = nn.Sequential(
            nn.Linear(width, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Input projection
        self.input_proj = nn.Linear(input_channels, width)
        
        # Transformer blocks
        # First half: self-attention only
        self.blocks = nn.ModuleList([
            TransformerBlock(
                width=width,
                heads=heads,
                context_dim=None,  # No cross-attention in first half
                skip_ln=skip_ln
            )
            for _ in range(layers // 2)
        ])
        
        # Middle block with cross-attention
        self.mid_block = TransformerBlock(
            width=width,
            heads=heads,
            context_dim=context_dim,
            skip_ln=skip_ln
        )
        
        # Second half: with cross-attention
        self.cross_blocks = nn.ModuleList([
            TransformerBlock(
                width=width,
                heads=heads,
                context_dim=context_dim,
                skip_ln=skip_ln
            )
            for _ in range(layers // 2)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(width, output_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Initialize output projection with zero weights for better training stability
        nn.init.zeros_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.LayerNorm)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, n_ctx, input_channels] noisy latents
            t: [B,] timesteps
            context: Optional [B, seq_len, context_dim] conditioning context
            
        Returns:
            [B, n_ctx, output_channels] predicted noise
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.width)
        t_emb = self.time_embed(t_emb)  # [B, time_embed_dim]
        
        # Project input
        h = self.input_proj(x)  # [B, n_ctx, width]
        
        # Add time embedding
        h = h + t_emb.unsqueeze(1)
        
        # First half: self-attention only
        for block in self.blocks:
            h = block(h, context=None)
        
        # Middle block with cross-attention
        h = self.mid_block(h, context)
        
        # Second half: with cross-attention
        for block in self.cross_blocks:
            h = block(h, context)
        
        # Output projection
        out = self.output_proj(h)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and cross-attention
    """
    def __init__(
        self,
        width: int,
        heads: int,
        context_dim: Optional[int] = None,
        skip_ln: bool = True
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-5)
        self.attn = MultiheadAttention(
            embed_dim=width,
            num_heads=heads,
            context_dim=width,  # Self-attention
            dropout=0.0
        )
        
        self.has_cross_attn = context_dim is not None
        if self.has_cross_attn:
            self.norm2 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-5)
            self.cross_attn = MultiheadAttention(
                embed_dim=width,
                num_heads=heads,
                context_dim=context_dim,  # Cross-attention
                dropout=0.0
            )
        
        self.norm3 = nn.LayerNorm(width, elementwise_affine=True, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
            nn.Dropout(0.1)  # Small dropout for regularization
        )
        
        # Skip layer norm
        self.skip_ln = skip_ln
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize MLP layers
        nn.init.xavier_uniform_(self.mlp[0].weight, gain=1e-10)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.xavier_uniform_(self.mlp[2].weight, gain=1e-10)
        nn.init.zeros_(self.mlp[2].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention with pre-norm
        h = self.norm1(x) if self.skip_ln else x
        h = self.attn(query=h, key=h, value=h)
        x = x + h
        
        # Cross attention if context is provided
        if self.has_cross_attn and context is not None:
            h = self.norm2(x) if self.skip_ln else x
            h = self.cross_attn(query=h, key=context, value=context)
            x = x + h
        
        # MLP with pre-norm
        h = self.norm3(x) if self.skip_ln else x
        h = self.mlp(h)
        x = x + h
        
        return x


class MultiheadAttention(nn.Module):
    """
    Multi-head attention with efficient implementation for both self and cross attention
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Ensure the embedding dimension is divisible by the number of heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        
        self.scaling = self.head_dim ** -0.5
        self.context_dim = context_dim or embed_dim
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.context_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.context_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        # Initialize projections like in the original transformer paper
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: [batch_size, seq_len, embed_dim]
            key: [batch_size, key_len, context_dim] or None (if self-attention)
            value: [batch_size, key_len, context_dim] or None (if self-attention)
            attention_mask: [batch_size, num_heads, seq_len, key_len] or None
            return_attn_weights: Whether to return attention weights
            
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attn_weights: Optional [batch_size, num_heads, seq_len, key_len]
        """
        batch_size, tgt_len, _ = query.size()
        
        # Use query as key/value if not provided (self-attention)
        if key is None:
            key = query
        if value is None:
            value = key
        
        src_len = key.size(1)
        
        # Project queries, keys, and values
        q = self.q_proj(query)  # [B, T, D]
        k = self.k_proj(key)    # [B, S, D]
        v = self.v_proj(value)  # [B, S, D]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [B, H, T, S]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)  # [B, H, T, D/H]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # [B, L, D]
        
        # Final projection
        output = self.out_proj(output)
        return output


def timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings with configurable precision.
    
    Args:
        timesteps: 1D tensor of timesteps of shape [batch_size,]
        dim: Dimension of the output embeddings
        max_period: Controls the minimum frequency of the embeddings
        dtype: Data type of the output tensor
        
    Returns:
        Tensor of shape [batch_size, dim] containing the timestep embeddings
    """
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"
    
    # Create position encodings
    half_dim = dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    ) / (half_dim - 1)
    exponent = exponent.to(dtype)
    
    # Compute frequencies
    freqs = torch.exp(exponent)
    
    # Compute arguments for sin and cos
    args = timesteps[:, None].to(dtype) * freqs[None, :]
    
    # Create sin and cos embeddings
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    
    # Ensure output has the correct shape and type
    assert embedding.shape == (timesteps.shape[0], dim)
    return embedding
