#!/usr/bin/bash
#SBATCH --job-name=hat-finetune
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=50g
#SBATCH -e runs/slurm-%j.err
#SBATCH -o runs/slurm-%j.out
#SBATCH --partition=gpu

# Activate conda environment

export CUDA_VISIBLE_DEVICES=1

echo $CUDA_VISIBLE_DEVICES

wandb login

source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

# Pass resume_from_checkpoint explicitly to train.py
accelerate launch train.py


# Cleanup
conda deactivate