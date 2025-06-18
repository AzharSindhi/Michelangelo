import torch.nn as nn
import json
from pointnet2_ops.pointnet2_modules import PointnetSAModule

class Encoder(nn.Module):
    """
    Configurable PointNet2 Encoder for point cloud feature extraction.
    It is composed of a series of Set Abstraction modules, configured via a dictionary.
    """
    def __init__(self, config):
        super().__init__()

        arch_config = config['architecture']
        self.input_channels = config['in_fea_dim']
        bn_flag = config['bn']
        use_xyz_flag = config['model.use_xyz']

        npoint_list = json.loads(arch_config['npoint'])
        radius_list = json.loads(arch_config['radius'])
        nsample_list = json.loads(arch_config['nsample'])
        # feature_dim in config: [sa1_hidden, sa1_out/sa2_hidden, sa2_out/sa3_hidden, ... , saN_out]
        # e.g., for 4 SA layers: [32, 64, 128, 256, 512]
        sa_mlp_config_dims = json.loads(arch_config['feature_dim'])

        self.sa_modules = nn.ModuleList()
        current_feature_dim = self.input_channels

        for i in range(len(npoint_list)):
            # MLP structure for SA_i: [current_feature_dim, hidden_i, hidden_i, out_i]
            # hidden_i is sa_mlp_config_dims[i]
            # out_i is sa_mlp_config_dims[i+1]
            mlp_spec = [
                current_feature_dim,
                sa_mlp_config_dims[i],
                sa_mlp_config_dims[i],
                sa_mlp_config_dims[i+1]
            ]
            
            self.sa_modules.append(
                PointnetSAModule(
                    npoint=npoint_list[i],
                    radius=radius_list[i],
                    nsample=nsample_list[i],
                    mlp=mlp_spec,
                    use_xyz=use_xyz_flag,
                    bn=bn_flag
                )
            )
            current_feature_dim = sa_mlp_config_dims[i+1]

    def forward(self, xyz, features):
        """
        Forward pass of the encoder.
        Args:
            xyz (torch.Tensor): (B, N, 3) tensor of point coordinates.
            features (torch.Tensor): (B, N, C) tensor of point features (if input_channels > 0).
                                      If input_channels is 0, this can be None or an empty tensor,
                                      as PointNetSAModule with use_xyz=True will use coordinates as features.
        Returns:
            list: A list of point coordinates at each sampling level (l_xyz[0] is input xyz).
            list: A list of point features at each sampling level (l_features[0] is input features).
        """
        # Ensure features is correctly handled if input_channels is 0
        # PointnetSAModule handles features=None if its input channel for MLP is 3 (from use_xyz=True)
        # and input_channels to Encoder was 0.
        # However, the first MLP layer in PointnetSAModule is defined as mlp[0] + 3 if use_xyz else mlp[0].
        # So, if self.input_channels = 0, features should be None or (B, N, 0)
        # and the first SA module's mlp[0] should be 0.
        # The current_feature_dim logic correctly sets mlp_spec[0] to self.input_channels for the first SA module.

        l_xyz, l_features = [xyz], [features]
        for module in self.sa_modules:
            new_xyz, new_features = module(l_xyz[-1], l_features[-1])
            l_xyz.append(new_xyz)
            l_features.append(new_features)
        return l_xyz, l_features
