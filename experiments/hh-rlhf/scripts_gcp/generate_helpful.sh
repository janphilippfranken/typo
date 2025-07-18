#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate typo
cd /home/jphilipp/research_projects/typo/experiments/experiment_2

export CUDA_VISIBLE_DEVICES=7

# dataset 
constitution_key="helpful"
dataset_dir="${constitution_key}-base"

# iteration 
iteration=5
batch_size=2000

if (( iteration % 2 == 0 )); then
    start_example=0
    max_example=2000
else
    start_example=10000
    max_example=12000
fi

echo "Start Example: $start_example"
echo "Max Example: $max_example"

# hyperparams
lr=1e-7
beta=0.0
epoch=0.5

# model
if [ "$iteration" -gt 0 ]; then
    prev_iteration=$(($iteration - 1))
    model_path="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-2/beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-0.4/epoch-${epoch}"
    download_dir="/home/jphilipp/research_projects/typo_files/trained_models/checkpoints-exp-2/beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-0.4/epoch-${epoch}"
else
    model_path="mistralai/Mistral-7B-v0.1"
    download_dir="/home/jphilipp/research_projects/typo_files/pretrained_models/Mistral-7B-v0.1"
fi

file_name="${constitution_key}-beta-${beta}-lr-${lr}-iteration-${iteration}-epoch-${epoch}-from-iteration-4-epoch-0.4"

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
    batch_size=$batch_size \
    output_dir="data/iteration_${iteration}"