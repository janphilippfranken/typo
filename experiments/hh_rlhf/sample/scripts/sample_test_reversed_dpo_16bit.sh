#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=64GB                         # Memory request
#SBATCH --cpus-per-task=24                  # Number of CPUs per task
#SBATCH --time=128:00:00                    # Time limit
#SBATCH --output=rlhf_reversed_test_dpo.out         
#SBATCH --error=rlhf_reversed_test_dpo.err           

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/sample

for run in {4..10}
do
    python sample_test_dpo_16bit.py \
    sampler.run_id=$run \
    sampler.seed=$run \
    sampler.dataset_version=rlhf_reversed_test \
    sampler.chosen=rejected \
    sampler.rejected=chosen \
    sampler.generation_prompt=generation_prompt_base_3 \
    sampler.evaluation_prompt=evaluation_prompt_base_2 \
    sampler.n_revisions=50 \
    sampler.constitution_batch_size=1 \
    sampler.eval_batch_size=10 \
    sampler.num_return_sequences=12 \
    data.split=test 
done