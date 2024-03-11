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

declare -a betas=(0.1 0.3 0.5)
declare -a lrs=(5e-7 1e-6)
declare -a iterations=(0 1 2)

generate() {
    local beta=$1
    local lr=$2
    local iteration=$3
    declare -a ckeys=("helpful" "harmless")

    # Setup for model and download_dir paths
    local model_path=""
    local download_dir=""

    if [[ "$iteration" -eq 0 ]]; then
        model_path="mistralai/Mistral-7B-v0.1"
        download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"
    else
        model_path="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-$((iteration-1))/epoch-0"
        download_dir="$model_path"
    fi

    local start_example=$((iteration % 2 == 0 ? 0 : 20))
    local batch_size=20
    local max_example=$((start_example + batch_size))

    for ckey in "${ckeys[@]}"; do
        local data_dir="${ckey}-base"
        local file_name="${ckey}-iteration-${iteration}-${lr}-${beta}.json"

        python generate.py \
        constitution_key=$ckey \
        file_name=$file_name \
        dataset.data_dir=$data_dir \
        model_config.model="$model_path" \
        model_config.download_dir="$download_dir" \
        start_example=$start_example \
        max_example=$max_example \
        batch_size=$batch_size
    done
}

train() {
    local beta=$1
    local lr=$2
    local iteration=$3

    # Generate filenames based on beta, lr, and iteration
    local helpful="helpful-iteration-${iteration}-${lr}-${beta}.json"
    local harmless="harmless-iteration-${iteration}-${lr}-${beta}.json"

    torchrun --nproc_per_node=4 train_typo.py \
    typo.beta=$beta \
    wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
    training.checkpoint_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}" \
    training.lr=$lr \
    data.hh_rlhf.helpful=$helpful \
    data.hh_rlhf.harmless=$harmless
}

merge() {
    local beta=$1
    local lr=$2
    local iteration=$3

    python merge.py \
    --state_dict "/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-0/model.pt" \
    --output_dir "/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-0"
}

evaluate() {
    local beta=$1
    local lr=$2
    local iteration=$3
    local model_path="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-0"

    python evaluate.py \
    --output_dir "results/responses" \
    --file_name "typo-model-iteration-${iteration}-${lr}-${beta}" \
    --model_config.model="$model_path" \
    --model_config.download_dir="$model_path"
}

# Main loop for the workflow
for iteration in "${iterations[@]}"; do
    for beta in "${betas[@]}"; do
        for lr in "${lrs[@]}"; do
            echo "Processing: Iteration $iteration, Beta $beta, LR $lr"

            # Generate data with either the base model or the model from the last iteration
            generate $beta $lr $iteration

            # For iteration 0, there's no need to train/merge/evaluate immediately after generation
            if [[ "$iteration" -gt 0 ]]; then
                # Train on the data generated in the previous step
                train $beta $lr $iteration
                
                # Merge the training results
                merge $beta $lr $iteration
                
                # Evaluate the newly trained model
                evaluate $beta $lr $iteration
            fi
        done
    done
done
