#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2

# export CUDA_VISIBLE_DEVICES=0,1,2,3

beta=0.0
lr=1e-7
iteration=4
base_dir="data/iteration_3"
epoch=0.3

torchrun --nproc_per_node=8 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-${epoch}" \
    training.checkpoint_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-2/beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-${epoch}" \
    training.lr=$lr \
    data_path="${base_dir}" \
    helpful="helpful-beta-0.0-lr-1e-7-iteration-3-epoch-0.3-from-iteration-2-epoch-0.2.json" \
    harmless="harmless-beta-0.0-lr-1e-7-iteration-3-epoch-0.3-from-iteration-2-epoch-0.2.json" \
    n_examples=1000