#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=15:00:00
#SBATCH --mail-user=azhar.hussian@fau.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=job_sitavae
#SBATCH --output=/home/hpc/iwnt/iwnt150h/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/Michelangelo/slurm_logs/%x_%j.err

conda activate michelo2
python train_sitavae.py