#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128                           # Memory request
#SBATCH --cpus-per-task=24                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=job_output.constitutions_rlhf_mistral.out         
#SBATCH --error=job_output.constitutions_rlhf_mistral.err           


source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf

# This will give us 100 constitutions, each comes with 100 training examples so -> 10k for fine-tuning.
for run in {1..11}
do
    python main.py sampler.run_id=$run
done