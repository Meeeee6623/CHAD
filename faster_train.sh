#!/usr/bin/bash
#SBATCH --job-name=hat-finetune
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=40g
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --time=24:00:00

#SBATCH --mail-type=ALL             #Send email on all job events
#SBATCH --mail-user=bsc32@duke.edu   #Send all emails to email_address

# Load required modules
module load cuda/11.8
module load anaconda3/2023.09

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hat

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_LEVEL=NVL

# Enable mixed precision training
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# # Create accelerate config if it doesn't exist
# if [ ! -f "accelerate_config.yaml" ]; then
#     cat > accelerate_config.yaml << EOL
# compute_environment: LOCAL_MACHINE
# distributed_type: MULTI_GPU
# downcast_bf16: 'no'
# gpu_ids: all
# machine_rank: 0
# main_training_function: main
# mixed_precision: bf16
# num_machines: 1
# num_processes: 2
# rdzv_backend: static
# same_needs: true
# EOL
# fi

# Run training with accelerate
accelerate launch train.py \
    --config config.yaml \
    --resume_from_checkpoint ${CHECKPOINT:-""}

# Cleanup
conda deactivate 