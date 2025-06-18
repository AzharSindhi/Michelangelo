import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(version_base=None, config_path="/home/ez48awud/Documents/implementations/Michelangelo/configs/aligned_shape_latents", config_name="pointnetvae-256")
def main(cfg: DictConfig):
    # Instantiate the model from the config
    pointnet_cfg = cfg.model.params.shape_module_cfg
    
    # use instantiae from config with model object
    # pointnet_cfg_dict = OmegaConf.to_container(pointnet_cfg, resolve=True)
    # model = PointNet2CloudCondition(pointnet_cfg)
    model = instantiate(pointnet_cfg)
    model.cuda()
    model.eval()

    print("Model instantiated successfully.")

    # Create dummy data
    batch_size = 2
    # The npoint is a string in the yaml, so we need to evaluate it
    num_points = pointnet_cfg.architecture.npoint[0]
    in_channels = pointnet_cfg.in_fea_dim

    pointcloud = torch.randn(batch_size, num_points, in_channels + 3).cuda()
    condition = torch.randn(batch_size, num_points, pointnet_cfg.get('partial_in_fea_dim', in_channels) + 3).cuda()
    label = torch.randint(0, pointnet_cfg.num_class, (batch_size,)).cuda()
    diffusion_steps = torch.randint(100, size=(batch_size,), device=pointcloud.device)  # t ~ U[T]
    class_index = torch.rand((3, 256, 256)).cuda()

    print(f"Input pointcloud shape: {pointcloud.shape}")
    print(f"Input condition shape: {condition.shape}")
    print(f"Input label shape: {label.shape}")

    # The model returns a tuple, but for this test we are interested in the first element
    output = model(pointcloud, condition, label=label, ts=diffusion_steps.view(batch_size, ), class_index=class_index)
    
    if isinstance(output, tuple):
        noise, condition_out = output
        print(f"Noise shape: {noise.shape}")
        print(f"Condition out shape: {condition_out.shape}")
    else:
        print(f"Output shape: {output.shape}")

if __name__ == '__main__':
    main()
