#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

beta=0.0
lr=1e-7
iteration=1
base_dir="data/sweep_v2"

export CUDA_VISIBLE_DEVICES=4,5,6,7

# Set the delay between each check for new files (in seconds)
delay=180  # 30 minutes

# Set the maximum execution time for each training run (in seconds)
max_execution_time=3600  # 1 hour

# Create a temporary file to store the list of processed files
processed_files_file="processed_file_check.txt"

# Initialize the list of processed files
if [ ! -f "$processed_files_file" ]; then
    touch "$processed_files_file"
fi

# Loop indefinitely
while true; do
    # Get the list of JSON files in the base directory
    json_files=("${base_dir}"/*.json)
    
    # Loop over each JSON file
    for file in "${json_files[@]}"; do
        # Check if the file has already been processed
        if ! grep -q "^$file$" "$processed_files_file"; then
            # Extract the epoch from the file name
            epoch=$(echo "${file}" | sed -E 's/.*-epoch-([0-9.]+)\.json/\1/')
            echo "Processing new file: ${file}, Epoch: ${epoch}"
            
            helpful="helpful-iteration-${iteration}-lr-${lr}-beta-${beta}-epoch-epoch-${epoch}.json"
            harmless="harmless-iteration-${iteration}-lr-${lr}-beta-${beta}-epoch-epoch-${epoch}.json"
            
            checkpoint_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-1/typo-beta-${beta}-${lr}-iteration-${iteration}-epoch-${epoch}"
            
            # Run the training script with a time limit
            echo "Starting the training run for epoch ${epoch} with a maximum execution time of ${max_execution_time} seconds..."
            timeout $max_execution_time torchrun --nproc_per_node=4 train_typo.py \
                typo.beta=$beta \
                wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-${epoch}" \
                training.checkpoint_dir="$checkpoint_dir" \
                training.lr=$lr \
                data_path="${base_dir}" \
                helpful="${helpful}" \
                harmless="${harmless}" \
                n_examples=1500
            
            # Mark the file as processed
            echo "$file" >> "$processed_files_file"
        fi
    done
    
    # Wait for the specified delay before checking for new files again
    echo "Waiting for ${delay} seconds before checking for new files..."
    sleep $delay
done