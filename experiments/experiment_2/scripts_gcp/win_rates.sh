#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --mem=2GB                          # Memory request
#SBATCH --cpus-per-task=1                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=win_rates.out
#SBATCH --error=win_rates.err


source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_1


echo "SEED"
epoch=0.1
test_file="evaluation-beta-0.0-lr-1e-7-iteration-1-epoch-epoch-${epoch}-temperature-0.0.json"
helpful_win_rates_file_name="1vs0-epoch-${epoch}-helpful_seed_2"
harmless_win_rates_file_name="1vs0-epoch-${epoch}-harmless_seed_2"

python win_rates.py test="$test_file" \
                    helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                    harmless_win_rates_file_name="$harmless_win_rates_file_name"
