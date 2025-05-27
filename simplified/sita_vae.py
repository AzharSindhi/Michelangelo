import pytorch_lightning as pl
from michelangelo.models.tsal.sal_perceiver import AlignedShapeLatentPerceiver
from michelangelo.models.tsal.clip_asl_module import CLIPAlignedShapeAsLatentModule
from michelangelo.models.asl_diffusion.asl_udt import ConditionalASLUDTDenoiser

class End2End(pl.LightningModule):
    def __init__(self):
        super().__init__()
        shape_model = AlignedShapeLatentPerceiver()
        self.pc_image_out = CLIPAlignedShapeAsLatentModule(shape_model=shape_model)
        self.asl_diffuser = ConditionalASLUDTDenoiser()

    def forward(self, surface, images, text):
        embed_outputs, shape_latents = self.pc_image_out(surface, images, text)

        # embed_outputs = {
        #     "image_embed": image_embed,
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
        
        
        