#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --output=/home/ez48awud/Documents/implementations/Michelangelo/slurm_logs/%x_%j.out
#SBATCH --error=/home/ez48awud/Documents/implementations/Michelangelo/slurm_logs/%x_%j.err


echo $CUDA_VISIBLE_DEVICES