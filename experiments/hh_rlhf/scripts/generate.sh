#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=256                           # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=48:00:00                     # Time limit
#SBATCH --output=job_output.rejected_0.out         
#SBATCH --error=job_output.rejected_1.err           


source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf

python main.py