#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128GB                         # Memory request
#SBATCH --cpus-per-task=26                  # Number of CPUs per task
#SBATCH --time=128:00:00                    # Time limit
#SBATCH --output=rlhf_test.out         
#SBATCH --error=rlhf_test.err           

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/sample

for run in {1..10}
do
    python sample_test.py \
    sampler.run_id=$run \
    sampler.seed=$run \
    sampler.dataset_version=rlhf_test \
    sampler.chosen=chosen \
    sampler.rejected=rejected \
    sampler.generation_prompt=generation_prompt_base_2 \
    sampler.evaluation_prompt=evaluation_prompt_base_3 \
    sampler.n_revisions=50 \
    sampler.constitution_batch_size=1 \
    sampler.eval_batch_size=10 \
    sampler.num_return_sequences=12 \
    model_generate=mixtral_7b_vllm \
    model_evaluate=mixtral_7b_vllm \
    model_generate.completion_config.temperature=0.5 \
    data.split=test 
done