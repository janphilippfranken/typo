#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:1                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=merge.out                  # Standard output log with job ID
#SBATCH --error=merge.err                   # Standard error log with job ID

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_1

iteration=0
beta=0.1
lr=1e-6        # /scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-0.1-1e-6-iteration-0/epoch-1

checkpoint_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-1"
output_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}"
echo $checkpoint_dir
state_dict="${checkpoint_dir}/model.pt"

python merge.py \
    state_dict="$state_dict" \
    output_dir="$output_dir"
