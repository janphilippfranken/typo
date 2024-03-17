#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_3

# export CUDA_VISIBLE_DEVICES=4,


accelerate launch --config_file /home/jphilipp/research_projects/typo/experiments/experiment_3/conf/accelerate/deepspeed.yaml train_dpo.py \
    wandb.name="dpo" \
    training_args.output_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-3/dpo-baseline" \