#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=128:00:00                    # Time limit
#SBATCH --output=rlhf_reversed_mixtral.out         
#SBATCH --error=rlhf_reversed_mixtral.err           

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/sampler

# Loop over 100 runs
for run in {2..20}
do
    python sampler.py \
    sampler.run_id=$run \
    sampler.seed=$run \
    sampler.dataset_version=rlhf_reversed \
    sampler.chosen=rejected \
    sampler.rejected=chosen \
    sampler.generation_prompt=generation_prompt_base_2 \
    sampler.evaluation_prompt=evaluation_prompt_base_2 \
    sampler.n_revisions=50 \
    sampler.constitution_batch_size=1 \
    sampler.eval_batch_size=10 \
    sampler.num_return_sequences=20 \
    model_generate.completion_config.temperature=0.6 \
    model_generate.model_config.tensor_parallel_size=2
done