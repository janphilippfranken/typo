#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:1                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=evaluate.out            # Standard output log with job ID
#SBATCH --error=evaluate.err             # Standard error log with job ID

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/summarization

iteration=1
beta=0.0
lr=1e-7
# model_path="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}"
# download_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}"

model_path="mistralai/Mistral-7B-v0.1"
download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"

python evaluate.py \
    start_example=0 \
    max_example=250 \
    batch_size=250 \
    model_config.model="$model_path" \
    model_config.download_dir="$download_dir" \
    file_name="evaluation-base"
