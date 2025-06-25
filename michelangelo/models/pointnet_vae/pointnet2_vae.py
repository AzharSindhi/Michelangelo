import torch
import torch.nn as nn

# from pointnet2.data import Indoor3DSemSeg
# from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
from michelangelo.models.pointnet_vae.pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
# from pointnet2_ssg_sem import PointNet2SemSegSSG, calc_t_emb, swish
# from clip_encoder import CLIPEncoder
# from dino_encoder import DinoEncoder
from michelangelo.models.pointnet_vae.pnet import Pnet2Stage
# from model_utils import get_embedder
from michelangelo.models.pointnet_vae.mlp_transform import ProjectCrossAttend
import torch.nn.functional as F
import copy
import numpy as np
from michelangelo.models.modules.distributions import DiagonalGaussianDistribution


# from pointnet2.models.Mink.Img_Encoder import ImageEncoder
# from pointnet2.models.Mink.attention_fusion import AttentionFusion

# ---
from math import pi, log
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F
from omegaconf import OmegaConf
from einops import rearrange, repeat
from typing import List

# helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        temp = self.to_kv(context)
        k,v = temp.chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# Convolutional Position Encoding
class ConvPosEnc(nn.Module):
    def __init__(self, dim_q, dim_content, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj_q = nn.Conv1d(
            in_channels=dim_q,
            out_channels=dim_q,
            kernel_size=k,
            stride=1,
            padding=k//2,
            groups=dim_q
        )

        self.proj_content = nn.Conv1d(
            in_channels=dim_content,
            out_channels=dim_content,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=dim_content
        )

    def forward(self,q,content):
        q = q.permute(0,2,1)
        q = self.proj_q(q) + q
        q = q.permute(0,2,1)

        # B,C,H,W = content.shape
        content = content.permute(0, 2, 1)
        content = self.proj_content(content) + content
        content = content.permute(0,2,1)

        return q,content

# main class
class AttentionFusion(nn.Module):
    def __init__(
        self,
        depth,                                  # Self-Attention deep
        dim,                                    # Q dim
        latent_dim = 512,                       # Content dim
        cross_heads = 1,                        # Cross-Attention Head
        latent_heads = 8,                       # Self-Attention Head
        cross_dim_head = 64,                    # Cross-Attention Head dim
        latent_dim_head = 64,                   # Self-Attention Head dim
        weight_tie_layers = False,
        pe=False
    ):
        super().__init__()

        self.pe = pe
        if(pe):
            # position encoding
            self.cpe = ConvPosEnc(
                dim_q=latent_dim,
                dim_content=dim
            )

        # Cross-Attention
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(latent_dim, dim, heads = cross_heads, dim_head = cross_dim_head), context_dim = dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])
        #
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        # Self-Attention
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(
        self,
        data,                           # Content data
        mask = None,                    # mask
        queries_encoder = None,         # Q data
    ):
        b, *_, device = *data.shape, data.device
        x = queries_encoder

        # ---- position encoding ----
        if(self.pe):
            x,data = self.cpe(
                q=x,
                content=data,
            )
        # ---- position encoding ----

        # ---- Cross-Attention----
        cross_attn, cross_ff = self.cross_attend_blocks
        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x
        # ---- Cross-Attention----


        #  ---- Self-Attention ----
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x
        #  ---- Self-Attention ----

        return x

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return swish(x)

class PointNorm(nn.Module):
    def __init__(self,dim,t=True):
        super().__init__()
        self.t = t
        self.norm = nn.BatchNorm1d(dim)
    def forward(self,x):
        if(self.t):
            x = x.permute(0,2,1)
            return self.norm(x).permute(0,2,1)
        return self.norm(x)

class PointNet2CloudCondition(PointNet2SemSegSSG):

    def _build_model(self):
        self.l_uvw = None
        self.encoder_cond_features = None
        self.decoder_cond_features = None
        self.global_feature = None

        # self.attention_setting = self.hparams.get("attention_setting", None)
        self.attention_setting = OmegaConf.to_container(self.hparams.get("attention_setting", None), resolve=True)
        self.FeatureMapper_attention_setting = copy.deepcopy(self.attention_setting)
        if self.FeatureMapper_attention_setting is not None:
            self.FeatureMapper_attention_setting['use_attention_module'] = (
                            self.FeatureMapper_attention_setting['add_attention_to_FeatureMapper_module'])

        self.global_attention_setting = self.hparams.get('global_attention_setting', None)

        self.bn = self.hparams.get("bn", True)
        self.scale_factor = 1
        self.record_neighbor_stats = self.hparams["record_neighbor_stats"]
        if self.hparams["include_class_condition"]:
            self.class_emb = nn.Embedding(self.hparams["num_class"], self.hparams["class_condition_dim"])

        in_fea_dim = self.hparams['in_fea_dim']
        partial_in_fea_dim = self.hparams.get('partial_in_fea_dim', in_fea_dim)
        self.attach_position_to_input_feature = self.hparams['attach_position_to_input_feature']
        if self.attach_position_to_input_feature:
            in_fea_dim = in_fea_dim + 3
            partial_in_fea_dim = partial_in_fea_dim + 3

        self.partial_in_fea_dim = partial_in_fea_dim
        self.include_abs_coordinate = self.hparams['include_abs_coordinate']
        self.pooling = self.hparams.get('pooling', 'max')

        self.network_activation = self.hparams.get('activation', 'relu')
        assert self.network_activation in ['relu', 'swish']
        if self.network_activation == 'relu':
            self.network_activation_function = nn.ReLU(True)
        elif self.network_activation == 'swish':
            self.network_activation_function = Swish()

        self.include_local_feature = self.hparams.get('include_local_feature', True)
        self.include_global_feature = self.hparams.get('include_global_feature', False)

        self.global_feature_dim = None
        remove_last_activation = self.hparams.get('global_feature_remove_last_activation', True)
        if self.include_global_feature:
            self.global_feature_dim = self.hparams['pnet_global_feature_architecture'][1][-1]
            self.global_pnet = Pnet2Stage(
                self.hparams['pnet_global_feature_architecture'][0],
                self.hparams['pnet_global_feature_architecture'][1],
                bn=self.bn,
                remove_last_activation=remove_last_activation
            )

        # ---- t_emb ----
        t_dim = self.hparams['t_dim']
        self.fc_t1 = nn.Linear(t_dim, 4*t_dim)
        self.fc_t2 = nn.Linear(4*t_dim, 4*t_dim)
        self.activation = swish
        # ---- t_emb ----

        self.map_type = self.hparams['map_type']
        self.condition_net_arch_outpoints = self.hparams['condition_net_architecture']['npoint'][-1]
        if self.include_local_feature:
            # build SA module for condition point cloud
            condition_arch = self.hparams['condition_net_architecture']
            npoint_condition = condition_arch['npoint']#[1024, 256, 64, 16]
            radius_condition = condition_arch['radius']#np.array([0.1, 0.2, 0.4, 0.8])
            nsample_condition = condition_arch['nsample']#[32, 32, 32, 32]
            feature_dim_condition = condition_arch['feature_dim']#[32, 32, 64, 64, 128]
            mlp_depth_condition = condition_arch['mlp_depth']#3
            self.SA_modules_condition = self.build_SA_model(
                npoint_condition,
                radius_condition,
                nsample_condition,
                feature_dim_condition,
                mlp_depth_condition,
                partial_in_fea_dim,
                False,
                False,
                neighbor_def=condition_arch['neighbor_definition'],
                activation=self.network_activation,
                bn=self.bn,
                attention_setting=self.attention_setting
            )


            # build feature transfer modules from condition point cloud to the noisy point cloud x_t at encoder
            mapper_arch = self.hparams['feature_mapper_architecture']
            encoder_feature_map_dim = mapper_arch['encoder_feature_map_dim']#[32, 32, 64, 64]


        # ---- Cross-Attention ----
        q = 512
        kv = 512
        self.att_c = AttentionFusion(
            dim=kv,  # the image channels
            depth=0,  # depth of net (self-attention - Processing的数量)
            latent_dim=q,  # the PC channels
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=32,  # number of dimensions per cross attention head
            latent_dim_head=6,  # number of dimensions per latent self attention head
            pe=False
        )
        self.att_noise = AttentionFusion(
            dim=q,  # the image channels
            depth=0,  # depth of net (self-attention - Processing的数量)
            latent_dim=kv,  # the PC channels
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=32,  # number of dimensions per cross attention head
            latent_dim_head=6,  # number of dimensions per latent self attention head
            pe=False
        )


        
        # build SA module for the noisy point cloud x_t
        arch = self.hparams['architecture']
        npoint = arch['npoint']#[1024, 256, 64, 16]
        radius = arch['radius']#[0.1, 0.2, 0.4, 0.8]
        nsample = arch['nsample']#[32, 32, 32, 32]
        feature_dim = arch['feature_dim']#[32, 64, 128, 256, 512]
        mlp_depth = arch['mlp_depth']#3
        # if first conv, first conv in_fea_dim + encoder_feature_map_dim[0] -> feature_dim[0]
        # if not first conv, mlp[0] = in_fea_dim + encoder_feature_map_dim[0]
        additional_fea_dim = encoder_feature_map_dim if(self.include_local_feature and self.map_type == "map_feature") else None
        self.SA_modules = self.build_SA_model(
            npoint,
            radius,
            nsample,
            feature_dim,
            mlp_depth,
            in_fea_dim+encoder_feature_map_dim[0] if(self.include_local_feature and self.map_type == "map_feature") else in_fea_dim,
            self.hparams['include_t'],
            self.hparams["include_class_condition"],
            include_global_feature=self.include_global_feature,
            global_feature_dim=self.global_feature_dim,
            additional_fea_dim = additional_fea_dim,
            neighbor_def=arch['neighbor_definition'],
            activation=self.network_activation,
            bn=self.bn,
            attention_setting=self.attention_setting,
            global_attention_setting=self.global_attention_setting)

        if self.include_local_feature:
            # build FP module for condition cloud
            include_grouper_condition = condition_arch.get('include_grouper', False)
            use_knn_FP_condition =  condition_arch.get('use_knn_FP', False)
            K_condition = condition_arch.get('K', 3)
            decoder_feature_dim_condition = condition_arch['decoder_feature_dim']#[32, 32, 64, 64, 128]
            decoder_mlp_depth_condition = condition_arch['decoder_mlp_depth']#3
            assert decoder_feature_dim_condition[-1] == feature_dim_condition[-1]
            self.FP_modules_condition = self.build_FP_model(
                decoder_feature_dim_condition,
                decoder_mlp_depth_condition,
                feature_dim_condition,
                partial_in_fea_dim,
                False,
                False,
                use_knn_FP=use_knn_FP_condition,
                K=K_condition,
                include_grouper = include_grouper_condition,
                radius=radius_condition,
                nsample=nsample_condition,
                neighbor_def=condition_arch['neighbor_definition'],
                activation=self.network_activation, bn=self.bn,
                attention_setting=self.attention_setting)

            # build mapper from condition cloud to input cloud at decoder
            decoder_feature_map_dim = mapper_arch['decoder_feature_map_dim']#[32, 32, 64, 64, 128]


        # build FP module for noisy point cloud x_t
        include_grouper = arch.get('include_grouper', False)
        use_knn_FP =  arch.get('use_knn_FP', False)
        K = arch.get('K', 3)
        decoder_feature_dim = arch['decoder_feature_dim']#[128, 128, 256, 256, 512]
        decoder_mlp_depth = arch['decoder_mlp_depth']#3
        assert decoder_feature_dim[-1] == feature_dim[-1]
        additional_fea_dim = decoder_feature_map_dim[1:] if(self.include_local_feature and self.map_type == "map_feature") else None
        # module_use_xyz = self.hparams.get("model.use_xyz", True)
        # use_attention_module = self.attention_setting.get("use_attention_module", True)
        self.FP_modules = self.build_FP_model(
            decoder_feature_dim,
            decoder_mlp_depth,
            feature_dim,
            in_fea_dim,
            self.hparams['include_t'],
            self.hparams["include_class_condition"],
            include_global_feature=self.include_global_feature,
            global_feature_dim=self.global_feature_dim,
            additional_fea_dim=additional_fea_dim,
            use_knn_FP=use_knn_FP,
            K=K,
            include_grouper = include_grouper,
            radius=radius,
            nsample=nsample,
            neighbor_def=arch['neighbor_definition'],
            activation=self.network_activation,
            bn=self.bn,
            attention_setting=self.attention_setting,
            global_attention_setting=self.global_attention_setting
        )
        # set back
        # self.hparams["model.use_xyz"] = module_use_xyz
        # self.attention_setting["use_attention_module"] = use_attention_module

        point_upsample_factor = self.hparams.get('point_upsample_factor', 1)
        if point_upsample_factor > 1:
            if self.hparams.get('include_displacement_center_to_final_output', False):
                point_upsample_factor = point_upsample_factor-1
            self.hparams['out_dim'] = int(self.hparams['out_dim'] * (point_upsample_factor+1))

        input_dim = decoder_feature_dim[0]+3
        if(self.include_local_feature and self.map_type == "map_feature"):
            input_dim = input_dim + decoder_feature_map_dim[0]

        self.fc_layer_noise = nn.Sequential(
            nn.Conv1d(input_dim + 3, 128, kernel_size=1, bias=self.hparams["bias"]),
            nn.GroupNorm(32, 128),
            copy.deepcopy(self.network_activation_function),
            nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
        )

        self.condition_loss = self.hparams["condition_loss"]
        if (self.include_local_feature and self.hparams["condition_loss"]):
            self.fc_layer_c = nn.Sequential(
                nn.Conv1d(input_dim + 3, 128, kernel_size=1, bias=self.hparams["bias"]),
                nn.GroupNorm(32, 128),
                copy.deepcopy(self.network_activation_function),
                nn.Conv1d(128, self.hparams['out_dim'], kernel_size=1),
            )

        self.image_fusion_strategy = self.hparams['image_fusion_strategy']
        self.use_cross_conditioning = self.hparams['use_cross_conditioning']
        # Initialize Image processor
        image_backbone = self.hparams["image_backbone"]
        self.class_names = self.hparams["clip_processor"]["class_names"]
        if image_backbone == "none" or self.image_fusion_strategy == "none":
            self.image_processor = None
            self.image_out_dim = 0
        elif image_backbone == "clip":
            self.image_processor = CLIPEncoder(class_names=self.class_names)
            self.image_out_dim = self.image_processor.out_dim
        elif image_backbone == "dino":
            self.image_processor = DinoEncoder()
            self.image_out_dim = self.image_processor.out_dim


        self.condition_net_arch_outpoints = self.hparams['condition_net_architecture']['npoint'][-1]
        self.condition_net_arch_outdim = self.hparams['condition_net_architecture']['feature_dim'][-1]

        if self.image_fusion_strategy == 'condition':
            self.conditon_img_transform = nn.Linear(self.image_out_dim + self.global_feature_dim, self.global_feature_dim)
        elif self.image_fusion_strategy == 'second_condition':
            self.conditon_img_transform = nn.Linear(self.image_out_dim + self.hparams["class_condition_dim"], self.hparams["class_condition_dim"])
        elif self.image_fusion_strategy == 'latent':
            self.cond_latent_transform = nn.Linear(self.image_out_dim + self.condition_net_arch_outdim, self.condition_net_arch_outdim)
            self.main_latent_transform = nn.Linear(self.image_out_dim + feature_dim[-1], feature_dim[-1])
        elif self.image_fusion_strategy == 'only_clip':
            self.cond_latent_transform = ProjectCrossAttend(feature_dim[-1], self.image_out_dim)
        elif self.image_fusion_strategy == "cross_attention":
            condition_feature_dims = self.hparams['condition_net_architecture']['feature_dim']
            self.condition_img_transform = nn.ModuleList([
                ProjectCrossAttend(feature_dim, self.image_out_dim)
                for feature_dim in condition_feature_dims[1:]
            ])
        
        decoder_feature_dim = arch['decoder_feature_dim']#[128, 128, 256, 256, 512]
        self.transformations = []
        for i in range(len(decoder_feature_dim) - 1):
            self.transformations.append(nn.Linear(decoder_feature_dim[i], decoder_feature_dim[i+1], device="cuda"))

    
    def reset_cond_features(self):
        # return
        self.l_uvw = None
        self.encoder_cond_features = None
        self.decoder_cond_features = None
        self.global_feature = None
    
    def cross_attend(self, features, image_features, cross_attnd_layer):
        return None
        transformed_img_features = self.img_feature_tranform[cross_attnd_layer](image_features)
        features = cross_attnd_layer(
            features.permute(0, 2, 1),
            queries_encoder=transformed_img_features
        ).permute(0, 2, 1).contiguous()
    
    def get_image_features(self, class_index):
        image_features = torch.zeros((class_index.shape[0], self.image_out_dim)).cuda()
        if class_index is None or self.image_fusion_strategy == 'none' or self.image_processor is None:
            print("Image fusion strategy is none or image processor is None")
            return image_features

        with torch.no_grad():
            image_features = self.image_processor.get_image_features(class_index)
            image_features = image_features / torch.norm(image_features, dim=1, keepdim=True)
        return image_features


    def _prepare_inputs(self, pointcloud, condition):
        """Prepare and preprocess input tensors."""
        
        with torch.no_grad():
            xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
            pointcloud = torch.cat([pointcloud, xyz_ori], dim=2)

            uvw_ori = condition[:,:,0:3] / self.scale_factor
            condition = torch.cat([condition, uvw_ori], dim=2)

            xyz, features = self._break_up_pc(pointcloud)
            xyz = xyz / self.scale_factor
            i_pc = pointcloud[:,:,3:6]

            uvw, cond_features = self._break_up_pc(condition)
            uvw = uvw / self.scale_factor
            
        return xyz, features, uvw, cond_features, i_pc
    
    def _prepare_pc_inputs(self, pointcloud):
        with torch.no_grad():
            xyz_ori = pointcloud[:,:,0:3] / self.scale_factor
            pointcloud = torch.cat([pointcloud, xyz_ori], dim=2)

            xyz, features = self._break_up_pc(pointcloud)
            xyz = xyz / self.scale_factor
            i_pc = pointcloud[:,:,3:6]
        return xyz, features, i_pc

    def _prepare_condition_inputs(self, condition):
        with torch.no_grad():
            uvw_ori = condition[:,:,0:3] / self.scale_factor
            condition = torch.cat([condition, uvw_ori], dim=2)

            uvw, cond_features = self._break_up_pc(condition)
            uvw = uvw / self.scale_factor
        return uvw, cond_features
     
    def _get_global_condition_embedding(self,i_pc):
        """Get time and class condition embeddings."""
        if self.include_global_feature:
            global_feature = self.global_pnet(i_pc.transpose(1,2))
            return global_feature
        else:
            return None


    def _apply_image_fusion(self, features, image_features):
        if self.image_fusion_strategy == 'condition':
            features = self.conditon_img_transform(features, image_features)
        elif self.image_fusion_strategy == 'second_condition':
            features = self.conditon_img_transform(features, image_features)
        elif self.image_fusion_strategy == 'latent':
            features = self.cond_latent_transform(features, image_features)
        elif self.image_fusion_strategy == 'only_clip':
            features = self.cond_latent_transform(features, image_features)
        elif self.image_fusion_strategy == 'cross_attention':
            features = self.condition_img_transform(features, image_features)
        return features

    def _bidirectional_cross_attention(self, features, cond_features):
        cond_features = self.att_c(
            features.permute(0, 2, 1),
            queries_encoder=cond_features.permute(0, 2, 1)
        ).permute(0, 2, 1).contiguous()

        features = self.att_noise(
                cond_features.permute(0, 2, 1),
                queries_encoder=features.permute(0, 2, 1)
            ).permute(0, 2, 1).contiguous()
        
        return features, cond_features
        
    def encode_main(self, xyz, features, t_emb=None, condition_emb=None, second_condition_emb=None):
        """Encode input point cloud and conditions into latent features."""
        l_xyz, l_features = [xyz], [features]
        
        # Encoder forward pass
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](
                l_xyz[i], l_features[i],
                t_emb=t_emb,
                condition_emb=condition_emb,
                second_condition_emb=second_condition_emb,
                subset=True,
                record_neighbor_stats=self.record_neighbor_stats,
                pooling=self.pooling
            )
            l_xyz.append(li_xyz)
            l_features.append(li_features)
        
        return l_xyz, l_features
    
    def decode_main(self, l_xyz, l_features, t_emb=None, condition_emb=None, second_condition_emb=None):
        """Decode latent features into point cloud."""
        # Decoder forward pass
        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            
            # make sure l_features[i] and l_features[i-1] have the same dimensions, use transformations
            # l_features[i-1] = self.transformations[i](l_features[i-1].permute(0, 2, 1)).permute(0, 2, 1)
            # Main decoder
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i],
                l_features[i - 1], l_features[i],
                t_emb=t_emb,
                condition_emb=condition_emb,
                second_condition_emb=second_condition_emb,
                record_neighbor_stats=self.record_neighbor_stats,
                pooling=self.pooling
            )
        
        return l_features
    
    def encode_cond(self, uvw, cond_features):
        """Encode conditions into latent features."""
        l_uvw, l_cond_features = [uvw], [cond_features]
        
        # Encoder forward pass
        for i in range(len(self.SA_modules_condition)):
            li_uvw, li_cond_features = self.SA_modules_condition[i](
                l_uvw[i], l_cond_features[i],
                t_emb=None, condition_emb=None,
                subset=True,
                record_neighbor_stats=self.record_neighbor_stats,
                pooling=self.pooling
            )
            l_uvw.append(li_uvw)
            l_cond_features.append(li_cond_features)
        
        return l_uvw, l_cond_features
    
    def decode_cond(self, l_uvw, l_cond_features, global_feature):
        """Decode latent features into conditions."""
        # Decoder forward pass
        for i in range(-1, -(len(self.FP_modules_condition) + 1), -1):
            l_cond_features[i - 1] = self.FP_modules_condition[i](
                l_uvw[i - 1], l_uvw[i],
                l_cond_features[i - 1], l_cond_features[i],
                t_emb=None, condition_emb=global_feature,
                record_neighbor_stats=self.record_neighbor_stats,
                pooling=self.pooling
            )
        
        return l_cond_features
    
    def encode_latents(self, complete_pointcloud, incomplete_pointcloud, return_latents: bool = False):
        """

        Args:
            complete_pointcloud (torch.FloatTensor): [bs, n, 3 + c]
            incomplete_pointcloud (torch.FloatTensor): [bs, n, 3 + c]
        Returns:
            x (torch.FloatTensor): [bs, projection_dim]
            shape_latents (torch.FloatTensor): [bs, m, d]
        """

        # Prepare inputs and get initial features
        xyz, feats, i_pc = self._prepare_pc_inputs(complete_pointcloud)
        uvw, cond_feats = self._prepare_condition_inputs(incomplete_pointcloud)
        global_feature = self._get_global_condition_embedding(i_pc)

        # Encode
        l_xyz, l_features = self.encode_main(xyz, feats, condition_emb=global_feature)
        l_xyz_cond, l_cond_features = self.encode_cond(uvw, cond_feats)

        # bidirectional cross attention
        # l_features[-1] = self.att_noise(
        #     l_cond_features[-1].permute(0, 2, 1),
        #     queries_encoder=l_features[-1].permute(0, 2, 1)
        # ).permute(0, 2, 1).contiguous()

        l_cond_features[-1] = self.att_c(
            l_features[-1].permute(0, 2, 1),
            queries_encoder=l_cond_features[-1].permute(0, 2, 1)
        ).permute(0, 2, 1).contiguous()
        
        l_features_out = self.decode_main(l_xyz, l_features, condition_emb=global_feature)[0]
        l_cond_features_out = self.decode_cond(l_xyz_cond, l_cond_features, global_feature)[0]


        # transpose to [bs, m, d]
        if return_latents:
            return global_feature, l_features_out.permute(0, 2, 1), l_cond_features_out.permute(0, 2, 1)
        else:
            return global_feature

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
    
    def get_cond_features(self, incomplete_pointcloud):
        uvw, cond_feats = self._prepare_condition_inputs(incomplete_pointcloud)
        global_feature = self._get_global_condition_embedding(incomplete_pointcloud)
        l_xyz_cond, l_cond_features = self.encode_cond(uvw, cond_feats)
        l_cond_features_out = self.decode_cond(l_xyz_cond, l_cond_features, global_feature)[0]
        return l_cond_features_out.permute(0, 2, 1)

    def decode(self, out_feature: List[torch.FloatTensor], l_cond_feature: List[torch.FloatTensor], incomplete_pointcloud: torch.FloatTensor):
        uvw, _ = self._prepare_condition_inputs(incomplete_pointcloud)
        out_feature = torch.cat([out_feature, uvw, incomplete_pointcloud], dim=-1)
        out = self.fc_layer_noise(out_feature.transpose(1,2)).permute(0,2,1) # reconstructed output
        
        out_cond_feature = torch.cat([l_cond_feature, uvw, incomplete_pointcloud], dim=-1)
        out_partial = self.fc_layer_c(out_cond_feature.transpose(1,2)).permute(0,2,1) # reconstructed output

        return out, out_partial
    
    def forward(
            self,
            complete_pointcloud,
            incomplete_pointcloud,
            image=None,
    ):
        # Prepare inputs and get initial features
        xyz, feats, uvw, cond_feats, i_pc = self._prepare_inputs(
            complete_pointcloud, incomplete_pointcloud
        )
        
        # Encode
        l_xyz, l_features = self.encode_main(
            xyz, feats,
        )
        l_xyz_cond, l_cond_features = self.encode_cond(
            uvw, cond_feats,
        )

        # bidirectional cross attention
        # l_cond_features[-1] = self.att_c(
        #     l_features[-1].permute(0, 2, 1),
        #     queries_encoder=l_cond_features[-1].permute(0, 2, 1)
        # ).permute(0, 2, 1).contiguous()

        l_features[-1] = self.att_noise(
                l_cond_features[-1].permute(0, 2, 1),
                queries_encoder=l_features[-1].permute(0, 2, 1)
            ).permute(0, 2, 1).contiguous()
        
        # Decode
        out_feature = self.decode_main(
            l_xyz_cond, l_features,
        )
        
        # Process outputs
        out_feature = torch.cat([out_feature.transpose(1,2), i_pc, xyz], dim=-1).permute(0,2,1)
        out = self.fc_layer_noise(out_feature).permute(0,2,1) # reconstructed output
        
        return out
