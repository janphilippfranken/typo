#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:8                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=64                  
#SBATCH --time=256:00:00                    
#SBATCH --output=generate.out         
#SBATCH --error=generate.err     

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/summarization

# iteration 
iteration=2
batch_size=5000

if (( iteration % 2 == 0 )); then
    start_example=0
    max_example=5000
else
    start_example=5000
    max_example=10000
fi

echo "Start Example: $start_example"
echo "Max Example: $max_example"

# hyperparams
lr=1e-7
beta=0.0
epoch=1.0

# model
if [ "$iteration" -gt 0 ]; then
    model_path="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-summarization/typo-1e-7-iteration-2-from-epoch-1.0/epoch-1.0/"
    download_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-summarization/typo-1e-7-iteration-2-from-epoch-1.0/epoch-1.0/"
else
    model_path="mistralai/Mistral-7B-v0.1"
    download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"
fi

file_name="lr-${lr}-iteration-${iteration}-epoch-${epoch}-from-epoch-1.0"

# generate 
python generate.py \
    iteration="${iteration}" \
    file_name="${file_name}" \
    model_config.model="${model_path}" \
    model_config.download_dir="${download_dir}" \
    start_example=$start_example \
    max_example=$max_example \
    batch_size=$batch_size \
    output_dir="data/iteration_${iteration}" \
    file_name="${file_name}.json"