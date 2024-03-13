#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --mem=6GB                          # Memory request
#SBATCH --cpus-per-task=1                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=win_rates.out
#SBATCH --error=in_rates.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_2/v2

python win_rates.py