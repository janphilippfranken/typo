#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                       
#SBATCH --mem=512GB                       
#SBATCH --cpus-per-task=64               
#SBATCH --time=256:00:00                    
#SBATCH --output=train_typo.out         
#SBATCH --error=train_typo.err           

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/hh-rlhf

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

beta=0.0
lr=5e-7
iteration=2
checkpoint_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/hh-rlhf-fixed/typo-${lr}-iteration-${iteration}-from-epoch-0.12-sanity-check-4-gpus"

torchrun --nproc_per_node=4 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/iteration_1" \
    helpful="iteration-1-epoch-0.12-fixed-epoch-mistral-human-constitution-helpful.json" \
    harmless="iteration-1-epoch-0.12-fixed-epoch-mistral-human-constitution-harmless.json" \
    n_examples=1000 