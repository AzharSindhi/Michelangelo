#!/bin/bash

# Point Cloud Generation from Images
# Example usage: ./scripts/inference/image2pc.sh --config configs/pc_reconstruction.yaml --ckpt checkpoints/pc_recon.ckpt --input example_data/image/car.jpg --output output/car_pc.ply

python infer_pc.py \
--config $1 \
--ckpt $2 \
--input $3 \
--output $4 \
--num_points 4096 \
--device cuda
