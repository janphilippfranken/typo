#!/bin/bash

# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

# Navigate to the experiment directory
cd /home/jphilipp/research_projects/typo/experiments/experiment_1

iteration=1
beta=0.0
lr=1e-7
checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/typo-beta-${beta}-${lr}-iteration-${iteration}"

shopt -s nullglob
for dir in ${checkpoint_base_dir}-epoch-*; do
    if [ -d "${dir}" ]; then
        # echo "Processing ${dir}"
        
        # Look for a directory named 'epoch-0.33' inside the current directory
        if [ -d "${dir}/epoch-0.33" ]; then
            echo "Found epoch-0.33 inside ${dir}"

            # Extract a unique identifier from $dir, e.g., the epoch value
            parent_epoch=$(basename "${dir}")

            # Construct the output directory with the unique identifier
            output_dir="/home/jphilipp/research_projects/typo_files/trained_models/merged-exp-1/iteration-1/${parent_epoch}"

            state_dict="${dir}/epoch-0.33/model.pt"

            # Check if the state_dict file exists
            if [ -f "$state_dict" ]; then
                echo $state_dict
                echo $output_dir
                # Note: Using the correct form with -- before the arguments
                python merge.py state_dict="$state_dict" output_dir="$output_dir"
                # No need to break if we want to process all matching directories
            else
                echo "The model state dict does not exist: ${state_dict}"
            fi
        fi
    fi
done
shopt -u nullglob
