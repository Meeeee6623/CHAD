#!/usr/bin/bash
#SBATCH --job-name=hat-inference # Changed job name
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=50g 
#SBATCH -e slurm-inference-%j.err # Changed log filename
#SBATCH -o slurm-inference-%j.out # Changed log filename
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate simpler_env

# --- Run Inference Script ---
# Specify the config file (usually config.yaml in the same directory)
# Specify the checkpoint directory you want to test
# Use --num_stories and --examples_per_story to control testing scope


python inference.py \
    --config config.yaml \
    --checkpoint /checkpoint/path/here \
    --num_stories 5 \
    --examples_per_story 6 \
    --memory_only

# Cleanup
conda deactivate

echo "Inference job finished."