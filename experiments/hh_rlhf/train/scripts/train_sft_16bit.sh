#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:3                        # Request GPUs
#SBATCH --mem=256GB                         # Memory request
#SBATCH --cpus-per-task=32                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=sft_16_bit.out
#SBATCH --error=sft_16_bit.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/train

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1

python train_sft_16bit.py \
    training_args.output_dir=/scr/jphilipp/scai/trained_models/Mixtral-8x7B-v0.1/checkpoints/16bit_sft \
    wandb.name=hh_rlhf_cai_16_bit_sft \