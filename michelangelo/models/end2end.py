import pytorch_lightning as pl
from michelangelo.models.tsal.sal_perceiver import AlignedShapeLatentPerceiver
from michelangelo.models.tsal.clip_asl_module import CLIPAlignedShapeAsLatentModule
from michelangelo.models.asl_diffusion.asl_udt import ConditionalASLUDTDenoiser
import torch

class End2End(pl.LightningModule):
    def __init__(self):
        super().__init__()
        shape_model = AlignedShapeLatentPerceiver(
            device="cuda",
            dtype=torch.float32,
            num_latents=256,
            embed_dim=64,
            point_feats=3,   # normal
            num_freqs=8,
            include_pi=False,
            heads=12,
            width=768,
            num_encoder_layers=8,
            num_decoder_layers=16,
            use_ln_post=True,
            init_scale=0.25,
            qkv_bias=False,
            use_checkpoint=True
        )

        self.pc_image_out = CLIPAlignedShapeAsLatentModule(shape_model=shape_model)
        self.asl_diffuser = ConditionalASLUDTDenoiser(
            device="cuda",
            dtype=torch.float32,
            input_channels=64,
            output_channels=64,
            n_ctx=256,
            width=768,
            layers=6,
            heads=12,
            context_dim=1024,
            init_scale=1.0,
            skip_ln=True,
            use_checkpoint=True
        )

    def forward(self, surface, images, text=None):
        embed_outputs, shape_latents = self.pc_image_out(surface, images, text)
        # shape_latents are other local queries of shape (batch_size, num_latents, embed_dim) used in diffusion model and decoder

        # embed_outputs = {
        #     "image_embed": image_embed, # this is a global token used in contrastive loss
        #     "text_embed": text_embed,
        #     "shape_embed": shape_embed,
        #     "logit_scale": self.clip_model.logit_scale.exp()
        # }
    
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass
    
    def configure_optimizers(self):
        pass
        
        
        