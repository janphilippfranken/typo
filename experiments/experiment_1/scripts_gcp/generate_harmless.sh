#!/bin/bash

source /opt/conda/etc/profile.d/conda.sh
conda activate typo

cd /home/jphilipp/research_projects/typo/experiments/experiment_1

# dataset 
constitution_key="harmless"
dataset_dir="${constitution_key}-base"

# iteration 
iteration=0
batch_size=10000

if (( iteration % 2 == 0 )); then
    start_example=0
    max_example=10000
else
    start_example=10000
    max_example=20000
fi


echo "Start Example: $start_example"
echo "Max Example: $max_example"

# hyperparams
lr=1e-7
beta=0.0

export CUDA_VISIBLE_DEVICES=2,3

# model
if [ "$iteration" -gt 0 ]; then
    prev_iteration=$(($iteration - 1))
    model_path="/home/jphilipp/research_projects/typo_files/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${prev_iteration}"
    download_dir="/home/jphilipp/research_projects/typo_files/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${prev_iteration}"
else
    model_path="mistralai/Mistral-7B-v0.1"
    download_dir="/home/jphilipp/research_projects/typo/pretrained_models/Mistral-7B-v0.1"
fi


# filename for sweep
file_name="${constitution_key}-iteration-${iteration}-lr-${lr}-beta-${beta}"

# generate 
python generate.py \
    iteration="${iteration}" \
    constitution_key="${constitution_key}" \
    file_name="${file_name}" \
    dataset.data_dir="${dataset_dir}" \
    model_config.model="${model_path}" \
    model_config.download_dir="${download_dir}" \
    start_example=$start_example \
    max_example=$max_example \
    batch_size=$batch_size