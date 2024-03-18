#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_3

export CUDA_VISIBLE_DEVICES=0

checkpoint_base_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-3/dpo-baseline/checkpoint-936/"



python evaluate.py \
  start_example=0 \
  max_example=250 \
  batch_size=250 \
  model_type="dpo" \
  model_config.model="$checkpoint_base_dir" \
  model_config.download_dir="$checkpoint_base_dir" \
  file_name="evaluation-dpo-baseline"
    



# # Create or clear the file that tracks processed epochs
# touch "$processed_epochs_file"

# while true; do
#     # Loop over each subdirectory in checkpoint_base_dir, assuming they represent epochs
#     for epoch_dir in ${checkpoint_base_dir}beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-0.1/epoch-*; do
#         if [ -d "$epoch_dir" ]; then
#             epoch=$(basename "$epoch_dir")
#             echo "Processing epoch: $epoch"
            
#             # Check if epoch has already been processed
#             if ! grep -q "$epoch" "$processed_epochs_file"; then
#                 model_path="${epoch_dir}"
#                 download_dir="${epoch_dir}"

#                 echo "Evaluating for epoch: $epoch with iteration: $iteration"

#                 python evaluate.py \
#                     start_example=0 \
#                     max_example=100 \
#                     batch_size=100 \
#                     model_config.model="$model_path" \
#                     model_config.download_dir="$download_dir" \
#                     file_name="evaluation-beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-${epoch}-from-iteration-1-epoch-0.1"

#                 # Mark epoch as processed
#                 echo "$epoch" >> "$processed_epochs_file"
#             else
#                 echo "Epoch $epoch has already been evaluated."
#             fi
#         fi
#     done
    
#     # Wait for a specified amount of time (e.g., 1 hour) before checking again
#     echo "Waiting for new epochs..."
#     sleep 20
# done
