import os
import subprocess
from itertools import product
from tqdm import tqdm
import logging 

# Define the parameters
betas = [0.1]
lrs = [1e-6]
iterations = [0, 1, 2]

# Set the paths
base_model_path = "mistralai/Mistral-7B-v0.1"
base_download_dir = "/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"

def generate(iteration, ckey, file_name, model_path, download_dir, start_example, max_example):
    subprocess.run([
        "python", "generate.py",
        f"iteration={iteration}",
        f"constitution_key={ckey}",
        f"file_name={file_name}",
        f"dataset.data_dir={ckey}-base",
        f"model_config.model={model_path}",
        f"model_config.download_dir={download_dir}",
        f"start_example={start_example}",
        f"max_example={max_example}",
        "batch_size=3500"
    ], check=True)

def train(beta, lr, iteration):
    checkpoint_dir = f"/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-{beta}-{lr}-iteration-{iteration}"
    helpful_file = f"helpful-iteration-{iteration}-{lr}-{beta}.json"
    harmless_file = f"harmless-iteration-{iteration}-{lr}-{beta}.json"
    subprocess.run([
        "torchrun", "--nproc_per_node=4", "train_typo.py",
        f"typo.beta={beta}",
        f"wandb.name=typo-beta-{beta}-lr-{lr}-iteration-{iteration}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"training.lr={lr}",
        f"helpful={helpful_file}",
        f"harmless={harmless_file}"
    ], check=True)

def merge(beta, lr, iteration):
    state_dict = f"/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/checkpoints-exp-1-sweep/typo-beta-{beta}-{lr}-iteration-{iteration}/epoch-1/model.pt"
    output_dir = f"/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-{beta}-{lr}-iteration-{iteration}/epoch-1"
    subprocess.run([
        "python", "merge.py",
        f"state_dict={state_dict}",
        f"output_dir={output_dir}"
    ], check=True)

def evaluate(iteration, beta, lr, model_path, download_dir):
    subprocess.run([
        "python", "evaluate.py",
        "start_example=0",
        "max_example=1000",
        "batch_size=1000",
        "output_dir=results/responses",
        f"file_name=typo-model-iteration-{iteration}-{lr}-{beta}",
        f"model_config.model={model_path}",
        f"model_config.download_dir={download_dir}"
    ], check=True)


def main():
    logging.info("started iteration")
    for beta in betas:
        logging.info("beta", lr)
        for lr in lrs:
            logging.info("lr", lr)
            for iteration in tqdm(iterations, desc="iterating"):
                logging.info("iteration", iteration)
                model_path = base_model_path
                download_dir = base_download_dir

                if iteration > 0:
                    model_path = f"/scr/jphilipp/typo/trained_models/Mistral-7B-v0.1/merged-exp-1-sweep/typo-beta-{beta}-{lr}-iteration-{iteration-1}/epoch-1"
                    download_dir = model_path

                logging.info(f"Starting Iteration: {iteration} for Beta: {beta}, LR: {lr}")

                start_example = 0 if iteration % 2 == 0 else 200
                max_example = start_example + 200

                for ckey in ["helpful", "harmless"]:
                    logging.info(ckey)
                    file_name = f"{ckey}-iteration-{iteration}-{lr}-{beta}"
                    generate(iteration, ckey, file_name, model_path, download_dir, start_example, max_example)

                logging.info('training')
                train(beta, lr, iteration)
                logging.info('merge')
                merge(beta, lr, iteration)
                logging.info('eval')
                evaluate(iteration, beta, lr, model_path, download_dir)
                
                
if __name__ == "__main__":
    main()