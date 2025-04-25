#!/usr/bin/bash
#SBATCH --job-name=hat-finetune
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=50g
#SBATCH -e runs/slurm-%j.err
#SBATCH -o runs/slurm-%j.out
#SBATCH --partition=athena-genai
#SBATCH --account=pl217
#SBATCH --nodelist=node6


# Activate conda environment

export CUDA_VISIBLE_DEVICES=1

echo $CUDA_VISIBLE_DEVICES

wandb login 2d7ef6b48d8585e96e62fd153e23f1a90548cde2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

# Pass resume_from_checkpoint explicitly to train.py
accelerate launch --config-file accelerate_1gpu.yaml train.py
#torchrun --nproc_per_node=8 main.py +name=test_run experiment=video_generation 'load="/home/pl217/1X-Challenge-Duke/outputs/2025-03-20/14-15-51/checkpoints/epoch=21-step=600000.ckpt"' dataset=onex algorithm=dfot_video_pose @diffusion/continuous algorithm.checkpoint.strict=False


#

# Cleanup
conda deactivate

