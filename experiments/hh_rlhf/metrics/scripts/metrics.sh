#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128GB                         # Memory request
#SBATCH --cpus-per-task=24                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=metrics.out             # Customized output file name
#SBATCH --error=metrics.err              # Customized error file name

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/metrics

declare -a models=(
    "mixtral_7b_dpo_16bit"
    "mixtral_7b_base" 
    
)

for run in {1..20}; do
    for model in "${models[@]}"; do
        python metrics.py \
        model="$model" \
        constitution_file="rlhf_test_mixtral_7b_base_run_${run}"
    done
done
