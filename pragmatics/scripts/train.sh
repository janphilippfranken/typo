#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:3                        # Request GPUs
#SBATCH --mem=256GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=ppo3gpu.out
#SBATCH --error=ppo3gpu.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/pragmatics

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

declare -a betas=(0.6 0.7 0.8)
declare -a temperatures=(1)

for beta in "${betas[@]}"; do
    for temperature in "${temperatures[@]}"; do
        torchrun --nproc_per_node 3 train.py \
        ppo.beta=$beta \
        ppo.temperature=$temperature \
        wandb.name="hh-ppo-beta-${beta}-batch-size-120" \
        training.checkpoint_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/ppo-beta-${beta}-batch-size-120"
    done
done