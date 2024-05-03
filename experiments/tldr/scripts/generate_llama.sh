#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:4                      
#SBATCH --mem=512GB                       
#SBATCH --cpus-per-task=64               
#SBATCH --time=256:00:00                    
#SBATCH --output=generate.out         
#SBATCH --error=generate.err     

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/tldr

# generate 
python generate_diverse.py 