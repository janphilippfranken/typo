#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=36                  
#SBATCH --time=256:00:00                    
#SBATCH --output=typo-multi.out
#SBATCH --error=typo-multi.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/experiments

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

declare -a betas=(0.1)

for beta in "${betas[@]}"; do
    torchrun --nproc_per_node 4 train_typo_multi_turn.py \
    pragpo.beta=$beta \
    wandb.name="typo-beta-${beta}-iteration-1" \
    training.checkpoint_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints-exp-1/typo-beta-${beta}-iteration-1-multi-turn"
done
