import torch
import json
from example_pointnet_vae import PointNet2CloudCondition
from omegaconf import OmegaConf

def main():
    # Load configuration
    with open('pointnet_config.json', 'r') as f:
        config = json.load(f)

    # Instantiate the model
    model = PointNet2CloudCondition(OmegaConf.create(config))
    model.cuda()
    model.eval()

    print("Model instantiated successfully.")

    # Create dummy data
    batch_size = 2
    num_points = config['architecture']['npoint'][0]
    in_channels = config['in_fea_dim']

    pointcloud = torch.randn(batch_size, num_points, in_channels + 3).cuda()
    condition = torch.randn(batch_size, num_points, config.get('partial_in_fea_dim', in_channels) + 3).cuda()
    label = torch.randint(0, config['num_class'], (batch_size,)).cuda()
    class_index = torch.rand((3, 256, 256)).cuda()

    print(f"Input pointcloud shape: {pointcloud.shape}")
    print(f"Input condition shape: {condition.shape}")
    print(f"Input label shape: {label.shape}")

    noise, condition_out = model(pointcloud, condition, label=label, class_index=class_index)

    print(f"Noise shape: {noise.shape}")
    print(f"Condition out shape: {condition_out.shape}")

if __name__ == '__main__':
    main()
