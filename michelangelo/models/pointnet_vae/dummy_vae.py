import torch
import json
from encoder import Encoder
from decoder import Decoder

def run_dummy_vae():
    # 1. Load configuration
    config_path = 'pointnet_config.json'
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    
    # The actual config is nested under "pointnet_config" in the example JSON
    if 'pointnet_config' in config_data:
        config = config_data['pointnet_config']
    else:
        # Assuming the loaded JSON is directly the config if "pointnet_config" key is missing
        print(f"Warning: 'pointnet_config' key not found in {config_path}. Using the whole JSON as config.")
        config = config_data

    # 2. Initialize Encoder and Decoder
    encoder = Encoder(config=config)
    decoder = Decoder(config=config)
    print("Encoder and Decoder initialized successfully.")

    # 3. Create dummy input tensors
    batch_size = 2
    num_points = 2048  # Example number of points
    
    # XYZ coordinates (B, N, 3)
    dummy_xyz = torch.rand(batch_size, num_points, 3)

    # Features (B, N, C) where C is input_feature_dimension
    input_feature_dim = config.get('in_fea_dim', 0)
    if input_feature_dim > 0:
        dummy_features = torch.rand(batch_size, num_points, input_feature_dim)
    else:
        dummy_features = None 

    print(f"\nCreated dummy inputs:")
    print(f"  xyz shape: {dummy_xyz.shape}")
    if dummy_features is not None:
        print(f"  features shape: {dummy_features.shape}")
    else:
        print(f"  features: None (as input_feature_dim is {input_feature_dim})")

    # 4. Forward pass
    print("\nPerforming forward pass...")
    # Encoder pass
    l_xyz, l_features = encoder(dummy_xyz, dummy_features)
    print("Encoder output:")
    for i in range(len(l_xyz)):
        print(f"  Level {i} xyz shape: {l_xyz[i].shape}, features shape: {l_features[i].shape if l_features[i] is not None else 'None'}")

    # Decoder pass
    reconstructed_features = decoder(l_xyz, l_features)
    print("\nDecoder output (reconstructed features) shape:", reconstructed_features.shape)
    
    if reconstructed_features.shape[1] == num_points:
        print("Reconstruction successful: Number of points matches input.")
    else:
        print(f"Warning: Number of points in reconstruction ({reconstructed_features.shape[1]}) \
              does not match input ({num_points}).")

if __name__ == '__main__':
    run_dummy_vae()
