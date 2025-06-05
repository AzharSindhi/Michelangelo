#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=job_diff_scratch
#SBATCH --output=/home/woody/iwnt/iwnt150h/code/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/woody/iwnt/iwnt150h/code/Michelangelo/slurm_logs/%x_%j.err


conda activate michelo2
python train_diffusion.py -r scratch_mini_hpc_contrastive -m ./mlruns_hpc/mlruns