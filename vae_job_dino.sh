#!/bin/bash
#SBATCH --gres=gpu:3
#SBATCH --partition=48G
#SBATCH --time=2-00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=job_pointnetvae_dino_contrast
#SBATCH --output=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.err

conda activate michelo_poitnet
python train_sitavae.py -r pointnetvae_dino_contrast
