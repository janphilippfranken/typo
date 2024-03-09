#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --mem=64GB                           # Memory request
#SBATCH --cpus-per-task=48                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=harmless2.out
#SBATCH --error=harmless2.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

python generate_harmless_2.py