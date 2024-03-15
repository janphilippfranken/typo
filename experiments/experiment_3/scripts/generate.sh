#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:2                        # Request GPUs
#SBATCH --mem=64GB                          # Memory request
#SBATCH --cpus-per-task=8                   # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit hrs:min:sec
#SBATCH --output=generate.out      # Standard output log with job ID
#SBATCH --error=generate.err       # Standard error log with job ID

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_3/

# iteration 
iteration=0
batch_size=10000
start_example=0
max_example=20000

echo "Start Example: $start_example"
echo "Max Example: $max_example"

model_path="mistralai/Mistral-7B-v0.1"
download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"

# filename
file_name="data"

# generate 
python generate.py \
    iteration="${iteration}" \
    file_name="${file_name}" \
    model_config.model="${model_path}" \
    model_config.download_dir="${download_dir}" \
    start_example=$start_example \
    max_example=$max_example \
    batch_size=$batch_size