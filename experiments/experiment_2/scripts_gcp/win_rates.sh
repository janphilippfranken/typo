#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_2

model_dir="results/responses"
output_dir="results/win_rates"
processed_epochs_file="${output_dir}/processed_epochs.txt"

# Ensure the file exists
touch "$processed_epochs_file"

# Ensure the output directory exists
mkdir -p "$output_dir"

echo "Preparing for win rates processing..."

# Find and process all evaluation files for epochs
for test_file in ${model_dir}/evaluation-beta-0.0-lr-1e-7-iteration-1-epoch-*; do
    # Extracts the specific epoch number, taking into account the provided naming convention
    epoch=$(echo $test_file | grep -oP 'epoch-epoch-\K[\d\.]+(?=-temperature-0.0)')
    echo "Processing epoch: $epoch"
    test=$(basename "$test_file")
    echo "Test: $test"

    # Check if the epoch has been processed
    if ! grep -qx "epoch-${epoch}" "$processed_epochs_file"; then
        helpful_win_rates_file_name="1vs0-epoch-${epoch}-beta-0.0-lr-1e-7-iteration-1-helpful"
        harmless_win_rates_file_name="1vs0-epoch-${epoch}-beta-0.0-lr-1e-7-iteration-1-harmless"

        # Run win rates calculation with Hydra configuration syntax
        python win_rates.py test="$test" \
                            helpful_win_rates_file_name="$helpful_win_rates_file_name" \
                            harmless_win_rates_file_name="$harmless_win_rates_file_name"

        # Mark this epoch as processed
        echo "epoch-${epoch}" >> "$processed_epochs_file"
    else
        echo "Epoch $epoch has already been processed."
    fi
done

echo "Processing complete."
