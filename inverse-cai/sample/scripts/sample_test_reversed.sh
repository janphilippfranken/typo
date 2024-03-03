#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=86GB                         # Memory request
#SBATCH --cpus-per-task=26                  # Number of CPUs per task
#SBATCH --time=128:00:00                    # Time limit
#SBATCH --output=rlhf_reversed_test.out         
#SBATCH --error=rlhf_reversed_test.err           

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/sample

for run in {1..50}
do
    python sample_test.py \
    sampler.run_id=$run \
    sampler.seed=$run \
    sampler.dataset_version=rlhf_reversed_test \
    sampler.chosen=rejected \
    sampler.rejected=chosen \
    sampler.n_revisions=50 \
    sampler.constitution_batch_size=1 \
    sampler.eval_batch_size=10 \
    sampler.num_return_sequences=12 \
    model_generate.completion_config.temperature=0.5 \
    data.split=test 
done