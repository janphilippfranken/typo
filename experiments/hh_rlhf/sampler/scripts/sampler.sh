#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128GB                          # Memory request
#SBATCH --cpus-per-task=16                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=rlhf_reversed_mixtral_instruct_6.out         
#SBATCH --error=rlhf_reversed_mixtral_instruct_6.err           


source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/sampler

python sampler.py sampler.run_id=6
