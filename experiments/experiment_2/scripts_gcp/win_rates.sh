#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2


processed_epochs_file="${output_dir}/processed_epochs.txt"

# Ensure the file exists
touch "$processed_epochs_file"

model_dir="results/responses"

while true; do
    # Find all evaluation files for new epochs
    for test_file in ${model_dir}/evaluation-beta-0.0-lr-1e-7-iteration-1-epoch-epoch-*; do
        epoch=$(echo $test_file | grep -oP 'iteration-\K[^\-]+' | head -1) 

        # Check if the epoch has been processed
        if ! grep -q "$epoch" "$processed_epochs_file"; then
            test="evaluation-beta-1.0-lr-1e-6-iteration-${epoch}-temperature-0.0"
            helpful_win_rates_file_name="1vs${epoch}-beta-1.0-lr-1e-6-helpful"
            harmless_win_rates_file_name="1vs${epoch}-beta-1.0-lr-1e-6-harmless"

            # Assuming a command exists that takes these parameters and calculates win rates
            python win_rates.py test="$test" \
                                helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                                harmless_win_rates_file_name="$harmless_win_rates_file_name"
            
            echo "$epoch" >> "$processed_epochs_file"
        fi
    done
    
    echo "Waiting for new epochs..."
    sleep 3 
done
