#!/bin/bash
#SBATCH --gres=gpu:v100:4
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=job_diff_nocontrast
#SBATCH --output=/home/woody/iwnt/iwnt150h/code/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt150h/code/Michelangelo/slurm_logs/%x_%j.err


conda activate michelo2
python train_diffusion.py -r diffusion_dino_nocontrast_lrfixed -m ./mlruns_hpc/mlruns