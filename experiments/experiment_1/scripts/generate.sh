#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --mem=64GB                           # Memory request
#SBATCH --cpus-per-task=48                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=harmless3.out
#SBATCH --error=harmless3.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_1


python generate.py