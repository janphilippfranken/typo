#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:1                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=evaluate.out
#SBATCH --error=evaluate.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_1


python evaluate.py