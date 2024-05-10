#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:8              
#SBATCH --mem=812GB                       
#SBATCH --cpus-per-task=96              
#SBATCH --time=256:00:00                    
#SBATCH --output=train_llama.out         
#SBATCH --error=train_llama.err           

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/scale

# export MASTER_PORT=29501
# export MASTER_ADDR=cocoflops-hgx-1
# export CUDA_LAUNCH_BLOCKING=1

beta=0.0
lr=1e-6
iteration=2
checkpoint_dir="/scr/jphilipp/typo/trained_models/Meta-Llama-3-70B/checkpoints-diverse-ultra/typo-${lr}-iteration-${iteration}-from-epoch-0.4"

python train_llama.py \
    typo.beta=$beta \
    wandb.name="typo-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="training_data/base" \
    data_file="iteration_1_mix_ultra_harmless_1024_epoch_0.4.json" \
    n_examples=5120