#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the correct account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request 4 GPUs
#SBATCH --mem=256G                          # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=48:00:00                     # Time limit
#SBATCH --output=job_output.%j.out          # Standard output log
#SBATCH --error=job_output.%j.err           # Standard error log

# Your job commands follow here


# Load conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning


# Change to the directory with your Python script
cd ~/research_projects/scai-tuning/experiments/hh_rlhf

# Run
python main.py