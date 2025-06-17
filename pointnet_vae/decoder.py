import torch.nn as nn
import json
from pointnet2_ops.pointnet2_modules import PointnetFPModule

class Decoder(nn.Module):
    """
    Configurable PointNet2 Decoder for feature propagation.
    It is composed of a series of Feature Propagation modules to restore the original point cloud resolution.
    """
    def __init__(self, config):
        super().__init__()

        arch_config = config['architecture']
        self.input_channels = config['in_fea_dim'] # Corresponds to l_features[0]'s channel dim if not 0
        bn_flag = config['bn']

        # Encoder's per-level output feature dimensions
        # sa_mlp_config_dims: [sa1_hidden, sa1_out/sa2_hidden, ..., saN-1_out/saN_hidden, saN_out]
        sa_mlp_config_dims = json.loads(arch_config['feature_dim'])
        num_sa_layers = len(json.loads(arch_config['npoint']))
        
        # encoder_level_output_dims are the actual output dimensions of each SA layer
        # e.g., [sa1_out, sa2_out, sa3_out, sa4_out]
        # sa_mlp_config_dims[0] is hidden for SA1, sa_mlp_config_dims[1] is output of SA1
        encoder_level_output_dims = [sa_mlp_config_dims[i+1] for i in range(num_sa_layers)]

        # Decoder's per-level output feature dimensions for FP modules
        # decoder_fp_output_dims: [fp1_out, fp2_out, fp3_out, fp4_out, ...]
        # (Order corresponds to FP modules from deepest to shallowest if used directly)
        # Config: "decoder_feature_dim": "[128, 128, 256, 256, 512]"
        # This means: fp_out_for_level_1 (closest to input cloud) = 128
        #             fp_out_for_level_2 = 128
        #             fp_out_for_level_3 = 256
        #             fp_out_for_level_4 (deepest propagated) = 256
        decoder_fp_module_output_dims = json.loads(arch_config['decoder_feature_dim'])

        # There are num_sa_layers FP modules
        # FP modules propagate from deepest (SA_N output) to shallowest (original points)
        # self.fp_modules = nn.ModuleList()

        # FP4 (connects SA4 output and SA3 output)
        # Input to FP4: features from SA4 (l_features[4]) and SA3 (l_features[3])
        # Output dim of SA4: encoder_level_output_dims[3]
        # Output dim of SA3: encoder_level_output_dims[2]
        # Output dim of FP4: decoder_fp_module_output_dims[3] (using 0-indexed, so 4th element)
        self.fp4 = PointnetFPModule(mlp=[
            encoder_level_output_dims[3] + encoder_level_output_dims[2],
            decoder_fp_module_output_dims[3],
            decoder_fp_module_output_dims[3]
        ], bn=bn_flag)

        # FP3 (connects FP4 output and SA2 output)
        # Input to FP3: features from FP4_out and SA2 (l_features[2])
        # Output dim of FP4_out: decoder_fp_module_output_dims[3]
        # Output dim of SA2: encoder_level_output_dims[1]
        # Output dim of FP3: decoder_fp_module_output_dims[2]
        self.fp3 = PointnetFPModule(mlp=[
            decoder_fp_module_output_dims[3] + encoder_level_output_dims[1],
            decoder_fp_module_output_dims[2],
            decoder_fp_module_output_dims[2]
        ], bn=bn_flag)

        # FP2 (connects FP3 output and SA1 output)
        # Input to FP2: features from FP3_out and SA1 (l_features[1])
        # Output dim of FP3_out: decoder_fp_module_output_dims[2]
        # Output dim of SA1: encoder_level_output_dims[0]
        # Output dim of FP2: decoder_fp_module_output_dims[1]
        self.fp2 = PointnetFPModule(mlp=[
            decoder_fp_module_output_dims[2] + encoder_level_output_dims[0],
            decoder_fp_module_output_dims[1],
            decoder_fp_module_output_dims[1]
        ], bn=bn_flag)

        # FP1 (connects FP2 output and initial cloud features l_features[0])
        # Input to FP1: features from FP2_out and l_features[0]
        # Output dim of FP2_out: decoder_fp_module_output_dims[1]
        # Dim of l_features[0]: self.input_channels (or 0 if no initial features other than xyz)
        # Output dim of FP1: decoder_fp_module_output_dims[0]
        # Note: if self.input_channels is 0, l_features[0] might be None or (B,N,0).
        # PointnetFPModule handles this; its input channel count for skip_features is taken from the tensor itself.
        fp1_skip_connection_dim = self.input_channels
        self.fp1 = PointnetFPModule(mlp=[
            decoder_fp_module_output_dims[1] + fp1_skip_connection_dim,
            decoder_fp_module_output_dims[0],
            decoder_fp_module_output_dims[0]
        ], bn=bn_flag)

    def forward(self, l_xyz, l_features):
        """
        Forward pass of the decoder.
        Args:
            l_xyz (list): A list of point coordinates from the encoder.
                          l_xyz[0] = original_xyz, l_xyz[1]=sa1_xyz, ..., l_xyz[num_sa_layers]=saN_xyz.
            l_features (list): A list of point features from the encoder.
                               l_features[0] = original_features, l_features[1]=sa1_features, ...
        Returns:
            torch.Tensor: The final point features after propagation, corresponding to l_xyz[0].
        """
        # l_xyz: [xyz_in, xyz_sa1, xyz_sa2, xyz_sa3, xyz_sa4] (for 4 SA layers)
        # l_features: [feat_in, feat_sa1, feat_sa2, feat_sa3, feat_sa4]
        
        # Propagate features up. The indices for l_xyz and l_features correspond to:
        # Index 0: Input cloud level
        # Index 1: SA1 output level
        # Index 2: SA2 output level
        # Index 3: SA3 output level
        # Index 4: SA4 output level (deepest for a 4-SA-layer encoder)

        # Propagated features at SA3 level (from SA4 and SA3 features)
        feat_prop_sa3_lvl = self.fp4(l_xyz[3], l_xyz[4], l_features[3], l_features[4])
        # Propagated features at SA2 level (from propagated_SA3 and SA2 features)
        feat_prop_sa2_lvl = self.fp3(l_xyz[2], l_xyz[3], l_features[2], feat_prop_sa3_lvl)
        # Propagated features at SA1 level (from propagated_SA2 and SA1 features)
        feat_prop_sa1_lvl = self.fp2(l_xyz[1], l_xyz[2], l_features[1], feat_prop_sa2_lvl)
        # Final propagated features at input cloud level (from propagated_SA1 and initial features)
        final_features = self.fp1(l_xyz[0], l_xyz[1], l_features[0], feat_prop_sa1_lvl)

        return final_features
