#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo
cd /home/jphilipp/research_projects/typo/experiments/experiment_1

# dataset
constitution_key="harmless"
dataset_dir="${constitution_key}-base"
export CUDA_VISIBLE_DEVICES=2,3

# iteration
iteration=1
batch_size=5000
if (( iteration % 2 == 0 )); then
    start_example=0
    max_example=10000
else
    start_example=10000
    max_example=15000
fi
echo "Start Example: $start_example"
echo "Max Example: $max_example"

# hyperparams
lr=1e-7
beta=0.0

# Set the time limit (in seconds)
time_limit=2500  

# For iteration > 0, loop over all directories produced by the previous script
if [ "$iteration" -gt 0 ]; then
    prev_iteration=$(($iteration - 1))
    base_dir="/home/jphilipp/research_projects/typo_files/trained_models/merged-exp-1"

    # Iterate over each output directory from the previous step
    for model_dir in "${base_dir}"/*; do
        if [ -d "${model_dir}" ]; then
            # Ensure it's a directory
            epoch=$(basename "${model_dir}")
            echo "Processing model directory: ${model_dir}, Epoch: ${epoch}"
            model_path="${model_dir}"
            download_dir="${model_dir}"

            # filename for sweep, including epoch information
            file_name="${constitution_key}-iteration-${iteration}-lr-${lr}-beta-${beta}-epoch-${epoch}"

            # generate with timeout
            timeout "$time_limit" python generate.py \
                iteration="${iteration}" \
                constitution_key="${constitution_key}" \
                file_name="${file_name}" \
                dataset.data_dir="${dataset_dir}" \
                model_config.model="${model_path}" \
                model_config.download_dir="${download_dir}" \
                start_example=$start_example \
                max_example=$max_example \
                batch_size=$batch_size
        fi
    done
else
    model_path="mistralai/Mistral-7B-v0.1"
    download_dir="/home/jphilipp/research_projects/typo_files/pretrained_models/Mistral-7B-v0.1"

    # filename for sweep
    file_name="${constitution_key}-iteration-${iteration}-lr-${lr}-beta-${beta}"

    # generate with timeout
    timeout "$time_limit" python generate.py \
        iteration="${iteration}" \
        constitution_key="${constitution_key}" \
        file_name="${file_name}" \
        dataset.data_dir="${dataset_dir}" \
        model_config.model="${model_path}" \
        model_config.download_dir="${download_dir}" \
        start_example=$start_example \
        max_example=$max_example \
        batch_size=$batch_size
fi