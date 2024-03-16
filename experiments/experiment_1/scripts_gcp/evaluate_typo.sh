#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

export CUDA_VISIBLE_DEVICES=0
 
checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/iteration-"

iteration=1
beta=1.0
lr=1e-7

model_path="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/beta-${beta}-lr-${lr}-iteration-${iteration}/epoch-1"
download_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/beta-${beta}-lr-${lr}-iteration-${iteration}/epoch-1"

# model_path="mistralai/Mistral-7B-v0.1"
# download_dir="/home/jphilipp/research_projects/typo_files/pretrained_models/Mistral-7B-v0.1"

python evaluate.py \
    start_example=0 \
    max_example=500 \
    batch_size=500 \
    model_config.model="$model_path" \
    model_config.download_dir="$download_dir" \
    file_name="evaluation-beta-${beta}-lr-${lr}-iteration-${iteration}"