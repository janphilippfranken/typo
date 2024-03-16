#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

export CUDA_VISIBLE_DEVICES=3

checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/merged-exp-1/iteration-1"

# Infinite loop to run forever
while true; do
    for model_dir in "${checkpoint_base_dir}"/*; do

        # Extract the subdirectory name
        subdir=$(basename "${model_dir}")
        echo "Processing ${subdir}"
        
        model_path="${model_dir}"
        download_dir="${model_dir}"

        # Define the expected output file path
        output_file="${model_dir}/iteration-1-${subdir}.txt"

        # Check if the output file already exists
        if [ -f "$output_file" ]; then
            echo "Output file ${output_file} already exists, skipping..."
            continue
        fi

        echo "model_path: ${model_path}"

        python evaluate.py \
            start_example=0 \
            max_example=100 \
            batch_size=100 \
            model_config.model="$model_path" \
            model_config.download_dir="$download_dir" \
            file_name="iteration-1-${subdir}"
    done


done
