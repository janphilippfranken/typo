#!/bin/bash

#SBATCH --account=cocoflops                 # Specify the account
#SBATCH --partition=cocoflops               # Specify the partition
#SBATCH --nodelist=cocoflops-hgx-1          # Request the specific node
#SBATCH --gres=gpu:4                        # Request GPUs
#SBATCH --mem=312GB                         # Memory request
#SBATCH --cpus-per-task=64                  # Number of CPUs per task
#SBATCH --time=256:00:00                    # Time limit
#SBATCH --output=dpo-%j.out
#SBATCH --error=dpo-%j.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate scai-tuning

cd ~/research_projects/scai-tuning/pragmalign

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

declare -a betas=(0.1)
declare -a max_iters=(0)

for beta in "${betas[@]}"; do
    for max_iter in "${max_iters[@]}"; do
        accelerate launch --config_file conf/accelerate/deepspeed.yaml train_dpo_baseline.py \
        dpo.beta=$beta \
        wandb.name="dpo-beta-${beta}-iteration-1" \
        training_args.output_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/dpo-beta-${beta}-iteration-1-0-5k"
    done
done


