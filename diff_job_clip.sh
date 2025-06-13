#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=48G
#SBATCH --time=1-00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=job_diff_clip_nocontrast
#SBATCH --output=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.err

source ~/.bashrc
module load cuda
conda activate michelo
cd /home/ez48awud/azhar/code/Michelangelo/

python train_diffusion.py -r lrfixed1e-4 -e clipvsdino_nocontrast --use_clip_cond