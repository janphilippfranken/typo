#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=gpt3.out
#SBATCH --error=gpt3.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/pragmatics/hh_rlhf

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

python gpt4_win_rates.py