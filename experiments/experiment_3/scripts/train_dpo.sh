#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=48                  
#SBATCH --time=256:00:00                    
#SBATCH --output=dpo.out
#SBATCH --error=dpo.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_2/v2

accelerate launch --config_file /sailhome/jphilipp/research_projects/typo/experiments/experiment_2/v2/conf/accelerate/deepspeed.yaml train_dpo.py \
    wandb.name="dpo-no-sft" \
    training_args.output_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-2-v2/dpo-no-sft" \
    data_path="data" \
    helpful="helpful.json" \
    harmless="harmless.json" \
    n_examples=5000 