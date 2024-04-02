#!/bin/bash

#SBATCH --account=cocoflops                 
#SBATCH --partition=cocoflops              
#SBATCH --nodelist=cocoflops-hgx-1          
#SBATCH --gres=gpu:2                        
#SBATCH --mem=128GB                       
#SBATCH --cpus-per-task=18                  
#SBATCH --time=256:00:00                    
#SBATCH --output=generateharm.out         
#SBATCH --error=generateharm.err     

# Activate conda environment
source /scr/jphilipp/miniconda3/etc/profile.d/conda.sh
conda activate typo
    
cd ~/research_projects/typo/experiments/hh-rlhf

# iteration 
# generate 
python generate.py 
   