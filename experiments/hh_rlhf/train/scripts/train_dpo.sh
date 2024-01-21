#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128GB                         # Memory request
#SBATCH --cpus-per-task=26                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=dpo.out
#SBATCH --error=dpo.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/train

accelerate launch --config_file conf/accelerate/accelerate.yaml train_dpo.py \
    training_args.output_dir=/scr/jphilipp/scai/trained_models/Mixtral-8x7B-v0.1/cai_data_hh_rlhf_synthetic/checkpoints \
    wandb.name=cai_data_hh_rlhf_synthetic \
    data.cai_data=data/cai_data_hh_rlhf_synthetic