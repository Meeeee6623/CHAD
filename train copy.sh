#!/usr/bin/bash
#SBATCH --job-name=hat-finetune
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=50g
#SBATCH -e runs/slurm-%j.err
#SBATCH -o runs/slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --partition=athena-genai
#SBATCH --account=pl217
#SBATCH --nodelist=node6

# Activate conda environment

export CUDA_VISIBLE_DEVICES=1

echo $CUDA_VISIBLE_DEVICES

wandb login 2d7ef6b48d8585e96e62fd153e23f1a90548cde2 # shhhh

source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

# Pass resume_from_checkpoint explicitly to train.py
accelerate launch --config-file deep_hopfield_acc_config.yaml train.py --config config_hopfield_deep.yaml


# Cleanup
conda deactivate