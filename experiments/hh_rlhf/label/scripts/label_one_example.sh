#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=24                  # Number of CPUs per task
#SBATCH --time=128:00:00                    # Time limit
#SBATCH --output=label_one_example.out         
#SBATCH --error=label_one_example.err           

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/label

python label_one_example.py 