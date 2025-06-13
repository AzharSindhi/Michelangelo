#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --partition=middle,big
#SBATCH --time=2-00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=ALL
#SBATCH --job-name=job_vaes
#SBATCH --array=0-1
#SBATCH --output=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/ez48awud/azhar/code/Michelangelo/slurm_logs/%x_%j.err

source ~/.bashrc
module load cuda
conda activate michelo
cd /home/ez48awud/azhar/code/Michelangelo/

commands=(
    "python train_sitavae.py -r vae_contrast -e clipvsdino_sitavae --use_clip_cond --use_contrastive"
    "python train_sitavae.py -r vae_contrast -e clipvsdino_sitavae --use_contrastive"
)

eval ${commands[$SLURM_ARRAY_TASK_ID]}
