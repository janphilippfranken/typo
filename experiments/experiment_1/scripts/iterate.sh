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

declare -a betas=(0.1 0.5 1.0)
declare -a lrs=(1e-6)
declare -a iterations=(0 1 2)

for beta in "${betas[@]}"; do
    for lr in "${lrs[@]}"; do
        model_path="mistralai/Mistral-7B-v0.1" # use the base model for the first iteration
        download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"

        # eval
        python evaluate.py \
            start_example=0 \
            max_example=1000 \
            batch_size=1000 \
            output_dir="results/responses" \
            file_name="base-model" \
            model_config.model="mistralai/Mistral-7B-v0.1" \
            model_config.download_dir="/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1" 
        sleep 5

        for iteration in "${iterations[@]}"; do
            echo "Starting Iteration: $iteration for Beta: $beta, LR: $lr"

            if [[ "$iteration" -gt 0 ]]; then
                model_path="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-$((iteration-1))/epoch-1"
                download_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-$((iteration-1))/epoch-1"
                echo "Loaded Model Path: $model_path"
            fi

            start_example=$((iteration % 2 == 0 ? 0 : 10000))
            max_example=$((start_example + 10000))

            for ckey in "helpful" "harmless"; do
                data_dir="${ckey}-base"
                file_name="${ckey}-iteration-${iteration}-${lr}-${beta}"

                # generate
                python generate.py \
                iteration=$iteration \
                constitution_key=$ckey \
                file_name=$file_name \
                dataset.data_dir=$data_dir \
                model_config.model="$model_path" \
                model_config.download_dir="$download_dir" \
                start_example=$start_example \
                max_example=$max_example \
                batch_size=10000
            done
            sleep 5

            # train
            torchrun --nproc_per_node=4 train_typo.py \
                typo.beta=$beta \
                wandb.name="typo-beta-${beta}-lr-${lr}-iteration-${iteration}" \
                training.checkpoint_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}" \
                training.lr=$lr \
                helpful="helpful-iteration-${iteration}-${lr}-${beta}.json" \
                harmless="harmless-iteration-${iteration}-${lr}-${beta}.json"
            sleep 5

            # merge
            python merge.py \
                state_dict="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-1/model.pt" \
                output_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-1"
            sleep 5

            # eval
            python evaluate.py \
                start_example=0 \
                max_example=1000 \
                batch_size=1000 \
                output_dir="results/responses" \
                file_name="typo-model-iteration-${iteration}-${lr}-${beta}" \
                model_config.model="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-1" \
                model_config.download_dir="/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-${beta}-${lr}-iteration-${iteration}/epoch-1"
            sleep 5
        done
    done
done