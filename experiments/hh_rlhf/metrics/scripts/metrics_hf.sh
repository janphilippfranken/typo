#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:1                        # Request GPUs
#SBATCH --mem=64GB                         # Memory request
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=metrics3.out             # Customized output file name
#SBATCH --error=metrics3.err              # Customized error file name

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/metrics

declare -a models=(
    # "mixtral_7b_base" 
    "mixtral_7b_dpo_4bit"   
)


for run in {1..2}; do
    for model in "${models[@]}"; do
        python metrics_hf.py \
        model="mixtral_7b_base_hf" \
        constitution_file="rlhf_reversed_test_mixtral_7b_base_run_${run}"
    done
done
