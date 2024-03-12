#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit hrs:min:sec
#SBATCH --output=generate_harmless.out      # Standard output log with job ID
#SBATCH --error=generate_harmless.err       # Standard error log with job ID

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_1

# dataset 
constitution_key="harmless"
dataset_dir="${constitution_key}-base"

# iteration 
iteration=0
batch_size=4000

if (( iteration % 2 == 0 )); then
    start_example=0
    max_example=4000
else
    start_example=4000
    max_example=8000
fi


echo "Start Example: $start_example"
echo "Max Example: $max_example"

# hyperparams
lr=1e-6
beta=2.0

# model
if [ "$iteration" -gt 0 ]; then
    prev_iteration=$(($iteration - 1))
    model_path="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${prev_iteration}"
    download_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${prev_iteration}"
else
    model_path="mistralai/Mistral-7B-v0.1"
    download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"
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