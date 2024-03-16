#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/iteration-1"

for dir in ${checkpoint_base_dir}/epoch-*; do
    
    parent_epoch=$(basename "${dir}")



    output_dir="/home/jphilipp/research_projects/typo_files/trained_models/merged-exp-1/iteration-1/${parent_epoch}"
    
    # Use the looped directory for constructing state_dict path dynamically
    state_dict="${dir}/model.pt"

    echo "State dict: ${state_dict}"
    echo "Output dir: ${output_dir}"
    
    python merge.py state_dict="$state_dict" output_dir="$output_dir"

done
