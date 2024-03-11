#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --gres=gpu:4
#SBATCH --mem=312GB
#SBATCH --cpus-per-task=36
#SBATCH --time=256:00:00
#SBATCH --output=iterate.out
#SBATCH --error=iterate.err

source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo

cd ~/research_projects/typo/experiments/experiment_1

export MASTER_PORT=29501
export MASTER_ADDR=cocoflops-hgx-1
export CUDA_LAUNCH_BLOCKING=1

python iterate.py