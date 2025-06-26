#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --partition=48G
#SBATCH --time=2-00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=job_diff_dino_contrast
#SBATCH --output=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.err

conda activate michelo_poitnet

python train_diffusion.py -r pointnet_diff_dino_contrast