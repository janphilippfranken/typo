#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_3

export CUDA_VISIBLE_DEVICES=4,5,6,7


accelerate launch --config_file /home/jphilipp/research_projects/typo/experiments/experiment_3/conf/accelerate/deepspeed.yaml train_sft_typo.py \
    wandb.name="sft-both-typo" \
    training_args.output_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-3/sft-typo" \