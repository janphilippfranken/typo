#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:1                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=rlhf.out
#SBATCH --error=rlhf.err


source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/metrics

for run in {15..50}
do
    python metrics.py \
    constitution_file=rlhf_reverserd_gen_mixtral_7b_base_eval_mixtral_7b_base_gen_prompt_generation_prompt_base_2_run_$run \
    metrics.split=test
done