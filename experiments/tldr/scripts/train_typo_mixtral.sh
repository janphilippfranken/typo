#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:8                       
#SBATCH --mem=512GB                       
#SBATCH --cpus-per-task=96               
#SBATCH --time=256:00:00                    
#SBATCH --output=train_typo.out         
#SBATCH --error=train_typo.err           

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
checkpoint_dir="/scr/jphilipp/typo/trained_models/Mixtral-8x7b-v.01/checkpoints-sumarization-opus/typo-${lr}-iteration-${iteration}-opus"

python train_typo_mixtral.py \
    typo.beta=$beta \
    wandb.name="typo-lr-${lr}-iteration-${iteration}-opus" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/base" \
    data_file="base_mixtral_from_opus_principles.json" \
    n_examples=2000 

    