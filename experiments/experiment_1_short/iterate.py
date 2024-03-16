import subprocess
from tqdm import tqdm
import logging
from itertools import product

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define parameters
betas = [0.1]
lrs = [1e-6]
iterations = [0, 1, 2]

# Set paths
base_model_path = "mistralai/Mistral-7B-v0.1"
base_download_dir = "/scr/jphilipp/typo/pretrained_models/Mistral-7B-v0.1"

def run_command(command):
    """
    Utility function to run a subprocess command and ensure it executes successfully.
    """
    logging.info(f"Executing command: {' '.join(command)}")
    subprocess.run(command, check=True, text=True)

def generate(iteration, beta, lr, model_path, download_dir):
    """
    Function to call the generate.py script with appropriate parameters.
    """
    for ckey in ["helpful", "harmless"]:
        run_command([
            "python", "generate.py",
            f"iteration={iteration}",
            f"constitution_key={ckey}",
            f"file_name={ckey}-iteration-{iteration}-{lr}-{beta}",
            f"dataset.data_dir={ckey}-base",
            f"model_config.model={model_path}",
            f"model_config.download_dir={download_dir}",
            "start_example=0",
            "max_example=200",
            "batch_size=3500"
        ])

def train(beta, lr, iteration, checkpoint_dir):
    """
    Function to call the train_typo.py script with appropriate parameters.
    """
    run_command([
        "torchrun", "--nproc_per_node=4", "train_typo.py",
        f"typo.beta={beta}",
        f"wandb.name=typo-beta-{beta}-lr-{lr}-iteration-{iteration}",
        f"training.checkpoint_dir={checkpoint_dir}",
        f"training.lr={lr}"
    ])

def merge(beta, lr, iteration, checkpoint_dir):
    """
    Function to call the merge.py script with appropriate parameters.
    """
    state_dict = f"{checkpoint_dir}/epoch-1/model.pt"
    output_dir = f"{checkpoint_dir}/merged"
    run_command([
        "python", "merge.py",
        f"state_dict={state_dict}",
        f"output_dir={output_dir}"
    ])

def evaluate(iteration, beta, lr, model_path, download_dir):
    """
    Function to call the evaluate.py script with appropriate parameters.
    """
    run_command([
        "python", "evaluate.py",
        "start_example=0",
        "max_example=1000",
        "batch_size=1000",
        f"model_config.model={model_path}",
        f"model_config.download_dir={download_dir}",
        f"file_name=evaluation-iteration-{iteration}-{lr}-{beta}"
    ])

def main():
    for beta, lr, iteration in tqdm(list(product(betas, lrs, iterations)), desc="Overall Progress"):
        logging.info(f"Starting Iteration: {iteration} for Beta: {beta}, LR: {lr}")

        # Adjust model_path and download_dir based on iteration
        model_path = base_model_path if iteration == 0 else f"{base_download_dir}/merged-exp-1-sweep/typo-beta-{beta}-{lr}-iteration-{iteration-1}"
        download_dir = base_download_dir if iteration == 0 else model_path

        # Generate data
        generate(iteration, beta, lr, model_path, download_dir)

        # Define checkpoint directory for training
        checkpoint_dir = f"{base_download_dir}/checkpoints-exp-1-sweep/typo-beta-{beta}-{lr}-iteration-{iteration}"

        # Train model
        train(beta, lr, iteration, checkpoint_dir)

        # Merge model
        merge(beta, lr, iteration, checkpoint_dir)

        # Evaluate model
        evaluate(iteration, beta, lr, model_path, download_dir)

if __name__ == "__main__":
    main()
