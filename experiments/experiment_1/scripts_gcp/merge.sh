#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

iteration=0
beta=0.0
lr=1e-7

checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/typo-beta-${beta}-${lr}-iteration-${iteration}"

# Loop through each epoch directory in the checkpoint directory
for epoch_dir in "${checkpoint_base_dir}"/*; do
    if [ -d "${epoch_dir}" ]; then  # Make sure it's a directory
        echo "Processing ${epoch_dir}"
        output_dir="/home/jphilipp/research_projects/typo_files/trained_models/merged-exp-1/$(basename "${epoch_dir}")"
        state_dict="${epoch_dir}/model.pt"

        python merge.py \
            state_dict="$state_dict" \
            output_dir="$output_dir"
    fi
done
