#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=36                  
#SBATCH --time=256:00:00                    
#SBATCH --output=typo.out
#SBATCH --error=typo.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_2

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

declare -a betas=(0.1)

for beta in "${betas[@]}"; do
    torchrun --nproc_per_node 4 train_typo.py \
    pragpo.beta=$beta \
    wandb.name="typo-beta-${beta}-iteration-2" \
    training.checkpoint_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/typo-beta-${beta}-iteration-2-multiple-iterations"
done
