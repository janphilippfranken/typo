#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_3

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export MASTER_ADDR="localhost"
# export MASTER_PORT="12345"

beta=0.0
lr=1e-7
iteration=1
base_dir="data/base"
epoch=0.0

torchrun --nproc_per_node=4 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-3/typo/beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.lr=$lr \
    data_path="${base_dir}" \
    helpful="helpful-beta-0.0-lr-1e-7-iteration-0-epoch-0.0.json" \
    harmless="harmless-beta-0.0-lr-1e-7-iteration-0-epoch-0.0.json" \
    n_examples=1000