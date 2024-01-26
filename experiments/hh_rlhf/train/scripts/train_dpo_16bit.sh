#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:3                        # Request GPUs
#SBATCH --mem=256GB                         # Memory request
#SBATCH --cpus-per-task=36                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=dpo_16bit.out
#SBATCH --error=dpo_16bit.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/train

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1

python train_dpo_16bit.py \
    training_args.output_dir=/scr/jphilipp/scai/trained_models/Mixtral-8x7B-v0.1/checkpoints/16bit \
    wandb.name=cai_data_hh_rlhf_synthetic_16bit_rank_4 \
    data.cai_data=data/cai_data_hh_rlhf_synthetic