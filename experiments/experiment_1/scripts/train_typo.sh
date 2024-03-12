#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=36                  
#SBATCH --time=256:00:00                    
#SBATCH --output=train_typo.out         
#SBATCH --error=train_typo.err           

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/experiment_1

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

beta=2.0
lr=1e-6
iteration=0
checkpoint_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}"

torchrun --nproc_per_node=4 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="$checkpoint_dir" \
    training.lr=$lr \
    data_path="data/sweep" \
    helpful="helpful-iteration-${iteration}-lr-${lr}-beta-${beta}.json" \
    harmless="harmless-iteration-${iteration}-lr-${lr}-beta-${beta}.json" \
    n_examples=2000 