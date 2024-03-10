#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=48                  
#SBATCH --time=256:00:00                    
#SBATCH --output=sft.out
#SBATCH --error=sft.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_2/v2

accelerate launch --config_file /sailhome/jphilipp/research_projects/typo/experiments/experiment_2/v2/conf/accelerate/deepspeed.yaml train_sft.py \
    wandb.name="sft-iteration-1" \
    training_args.output_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-2-v2/sft-iteration-1"