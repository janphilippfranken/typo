#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=merge.out                  # Standard output log with job ID
#SBATCH --error=merge.err                   # Standard error log with job ID

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_2/v2

beta=1.0
lr=1e-6    

checkpoint_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-2-v2/sft-typo-both-beta-${beta}-${lr}/epoch-1"
output_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-2-v2/sft-typo-both-beta-${beta}-${lr}/epoch-1"
echo $checkpoint_dir
state_dict="${checkpoint_dir}/model.pt"

python merge.py \
    state_dict="$state_dict" \
    output_dir="$output_dir"
