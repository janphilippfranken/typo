#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2

export CUDA_VISIBLE_DEVICES=4,5,6,7

beta=0.0
lr=1e-7
iteration=1
base_dir="data/iteration_1"

torchrun --nproc_per_node=4 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-2/beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.lr=$lr \
    data_path="${base_dir}" \
    helpful="helpful.json" \
    harmless="harmless.json" \
    n_examples=1000