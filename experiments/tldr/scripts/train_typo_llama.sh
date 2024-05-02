#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                       
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=32               
#SBATCH --time=256:00:00                    
#SBATCH --output=train_llama.out         
#SBATCH --error=train_llama.err           

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/tldr

# export MASTER_PORT=29501
# export MASTER_ADDR=cocoflops-hgx-1
# export CUDA_LAUNCH_BLOCKING=1

beta=0.0
lr=5e-7
iteration=1
checkpoint_dir="/scr/jphilipp/typo/trained_models/Meta-Llama-3-8B/checkpoints-sumarization/typo-${lr}-iteration-${iteration}-opus-diverse"

python train_llama.py \
    typo.beta=$beta \
    wandb.name="typo-lr-${lr}-iteration-${iteration}-opus-diverse" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/base" \
    data_file="base_llama_from_opus_principles_diverse.json" \
    n_examples=2000 

    