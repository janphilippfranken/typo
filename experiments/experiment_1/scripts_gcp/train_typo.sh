#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# helpful="helpful-iteration-${iteration}-lr-${lr}-beta-${beta}.json" \
# harmless="harmless-iteration-${iteration}-lr-${lr}-beta-${beta}.json" \

beta=0.0
lr=1e-7
iteration=0
checkpoint_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/typo-beta-${beta}-${lr}-iteration-${iteration}"

torchrun --nproc_per_node=8 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/sweep_v2" \
    helpful="helpful.json" \
    harmless="harmless.json" \
    n_examples=5000 