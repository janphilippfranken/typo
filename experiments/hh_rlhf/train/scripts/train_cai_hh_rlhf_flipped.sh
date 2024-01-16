#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=128GB                         # Memory request
#SBATCH --cpus-per-task=26                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=train_cai_hh_rlhf_flipped.out
#SBATCH --error=train_cai_hh_rlhf_flipped.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments/hh_rlhf/train

accelerate launch --config_file conf/accelerate/accelerate.yaml --main_process_port 29501 train.py \
    training_args.output_dir=/scr/jphilipp/scai/trained_models/Mixtral-8x7B-v0.1/cai_data_hh_rlhf_flipped/checkpoints \
    wandb.name=cai_data_hh_rlhf_flipped \
    data.cai_data=data/cai_data_hh_rlhf_flipped