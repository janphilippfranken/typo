#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

iteration=0
beta=0.0
lr=1e-7
checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/merged-exp-1/iteration-${iteration}"

for model_dir in "${checkpoint_base_dir}"/*; do
    if [ -d "${model_dir}" ]; then
        # Make sure it's a directory
        echo "Processing ${model_dir}"

        # Extract the subdirectory name
        subdir=$(basename "${model_dir}")
        echo "Processing ${subdir}"
        
        model_path="${model_dir}"
        download_dir="${model_dir}"

        echo "model_path: ${model_path}"

        echo "evaluation-iteration-${iteration}-${lr}-${beta}-${subdir}"
        
        python evaluate.py \
            start_example=0 \
            max_example=500 \
            batch_size=500 \
            model_config.model="$model_path" \
            model_config.download_dir="$download_dir" \
            file_name="evaluation-iteration-${iteration}-${lr}-${beta}-${subdir}"
    fi
done