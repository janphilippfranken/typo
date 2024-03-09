#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                        
#SBATCH --mem=312GB                       
#SBATCH --cpus-per-task=48                  
#SBATCH --time=256:00:00                    
#SBATCH --output=dpo-%j.out
#SBATCH --error=dpo-%j.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/pragmalign

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

declare -a betas=(0.1)

for beta in "${betas[@]}"; do
    accelerate launch --config_file conf/accelerate/deepspeed.yaml train_dpo.py \
    dpo.beta=$beta \
    wandb.name="sft-positive-dpo-beta-${beta}-iteration-1" \
    training_args.output_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/sft-positive-dpo-beta-${beta}-iteration-1"
done