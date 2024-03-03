#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --mem=256GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=ppo-%j.out
#SBATCH --error=ppo-%j.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/pragmatics

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

declare -a betas=(0.1)
declare -a max_iters=(0)

for beta in "${betas[@]}"; do
    for max_iter in "${max_iters[@]}"; do
        torchrun --nproc_per_node 4 train.py \
        ppo.beta=$beta \
        ppo.loss="no_reference" \
        ppo.max_iter=$max_iter \
        wandb.name="beta-${beta}-iteration-2" \
        training.checkpoint_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints-no-icl/ppo-beta-${beta}-iteration-1-0-1k"
    done
done