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

source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

# Choose which configuration to use:
# 1. Deep Hopfield (Hopfield layers in each transformer block)
accelerate launch --config-file deep_hopfield_acc_config.yaml train.py --config config_hopfield_deep.yaml

# 2. Pre/Post Hopfield (Hopfield layers only at the beginning and end of the model)
# accelerate launch --config-file deep_hopfield_acc_config.yaml train.py --config config_hopfield_prepost.yaml

# Cleanup
conda deactivate