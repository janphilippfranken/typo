#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo
cd /home/jphilipp/research_projects/typo/experiments/experiment_1

beta=0.0
lr=1e-7
iteration=0
checkpoint_base_dir="/home/jphilipp/research_projects/typo/experiments/experiment_1/results/responses/sweep_2"
output_base_dir="/home/jphilipp/research_projects/typo/experiments/experiment_1/results/responses/sweep_2/win_rates"

while true; do
# Loop through each file in the checkpoint base directory
    for file in "${checkpoint_base_dir}"/evaluation-*.json; do
        if [ -f "${file}" ]; then
            # Make sure it's a file
            echo "Processing ${file}"

    
            file_name=$(basename "${file}")
            temperature=$(echo "$file_name" | sed -E 's/.*temperature-([0-9.]+)\.json/\1/')
            epoch=$(echo "$file_name" | sed -E 's/.*epoch-([0-9.]+)-temperature.*/\1/')
            
            test="evaluation-iteration-${iteration}-${lr}-${beta}-epoch-${epoch}-temperature-${temperature}"
            helpful_win_rates_file_name="typo-iteration-${iteration}vsbase-lr-${lr}-beta-${beta}-epoch-${epoch}-temperature-${temperature}-helpful"
            harmless_win_rates_file_name="typo-iteration-${iteration}vsbase-lr-${lr}-beta-${beta}-epoch-${epoch}-temperature-${temperature}-harmless"
            
            helpful_output_file="${output_base_dir}/${helpful_win_rates_file_name}.json"
            harmless_output_file="${output_base_dir}/${harmless_win_rates_file_name}.json"
            
            # Check if the output files already exist
            if [ -f "${helpful_output_file}" ] && [ -f "${harmless_output_file}" ]; then
                echo "Output files already exist. Skipping ${file}."
            else
                python win_rates.py \
                    test="$test" \
                    helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                    harmless_win_rates_file_name="$harmless_win_rates_file_name"
            fi
        fi
    done
    # Sleep for a certain interval before checking for new files again
    sleep 5
done
