#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

export CUDA_VISIBLE_DEVICES=1,2,3,4
# export MASTER_PORT=19501
# export MASTER_ADDR="127.0.0.2"

beta=1.0
lr=1e-7
iteration=2
base_dir="data/iteration_1"

torchrun --nproc_per_node=4 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.lr=$lr \
    data_path="${base_dir}" \
    helpful="helpful-beta-1.0-lr-1e-7-iteration-1.json" \
    harmless="harmless-beta-1.0-lr-1e-7-iteration-1.json" \
    n_examples=2000