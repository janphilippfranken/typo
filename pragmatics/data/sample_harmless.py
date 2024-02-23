import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

# Assuming the existence of PROMPT_CONTEXT, PROMPT_TRAINING, and other necessary constants
from prompts import PROMPT_CONTEXT, PROMPT_TRAINING
from seed_tasks import SEED_TASKS_HARMLESS

# Settings
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1"
CONSTITUTIONS_DIR = "constitutions"

# Load model
model = VLLMInferenceModel(
    model="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/sft-baseline/",
    download_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/sft-baseline",
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

# Load data
dataset = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train']

dataset = dataset.select(range(len(dataset) // 2, len(dataset)))  # only take the second half

formatted_train_data = {}

# Seeds for reproducibility
np.random.seed(1)
random.seed(1)

# Get a list of all constitution filenames without anti-constitutions
all_constitutions = [filename for filename in os.listdir(CONSTITUTIONS_DIR) if os.path.isfile(os.path.join(CONSTITUTIONS_DIR, filename))]

harmless_constitutions = [filename for filename in all_constitutions if "hh_h_h" in filename]
not_harmless_constitutions = [filename for filename in all_constitutions if "hh_h_not_h" in filename]

# Sort constitutions for consistency
harmless_constitutions = sorted(harmless_constitutions, key=extract_number)
not_harmless_constitutions = sorted(not_harmless_constitutions, key=extract_number)

for i, example in tqdm(enumerate(dataset), desc="Processing examples"):
    print(i)
    # Load seed example
    seed_example = np.random.choice(SEED_TASKS_HARMLESS)
    seed_example_chosen = seed_example['chosen'].strip()
    seed_example_rejected = seed_example['rejected'].strip()

    # Load conversation
    conversation = f"{remove_final_answer(example['chosen'])[0].strip()}"

    # Sample constitution
    random_constitution_index = np.random.randint(len(harmless_constitutions))
    harmless_constitution = json.load(open(os.path.join(CONSTITUTIONS_DIR, harmless_constitutions[random_constitution_index]), 'r'))['principles']
    not_harmless_constitution = json.load(open(os.path.join(CONSTITUTIONS_DIR, not_harmless_constitutions[random_constitution_index]), 'r'))['principles']
    random.shuffle(harmless_constitution)
    random.shuffle(not_harmless_constitution)
    harmless_constitution = f"1. {harmless_constitution[0]}\n2. {harmless_constitution[1]}"
    not_harmless_constitution = f"1. {not_harmless_constitution[0]}\n2. {not_harmless_constitution[1]}"

    # Format prompts
    prompt_harmless = PROMPT_CONTEXT.format(
        example_constitution_1=not_harmless_constitution,
        example_conversation_1=seed_example_rejected,
        example_constitution_2=harmless_constitution,
        example_conversation_2=seed_example_chosen,
        constitution=harmless_constitution,
        conversation=conversation,
    )

    prompt_not_harmless = PROMPT_CONTEXT.format(
        example_constitution_1=harmless_constitution,
        example_conversation_1=seed_example_chosen,
        example_constitution_2=not_harmless_constitution,
        example_conversation_2=seed_example_rejected,
        constitution=not_harmless_constitution,
        conversation=conversation,
    )

    prompts = [prompt_harmless, prompt_not_harmless]
    constitutions = [harmless_constitution, not_harmless_constitution]
    constitution_ids = [harmless_constitutions[random_constitution_index], not_harmless_constitutions[random_constitution_index]]

    responses = model.batch_prompt(
        prompts=prompts,
        max_new_tokens=200,
        top_p=0.9,
        temperature=0.5,
        num_return_sequences=1,
    )
    
    try:
        skip_example = not all(is_not_cut_off(response.strip()[-1]) for response in responses)

        if not skip_example:
            data = []
            for response, constitution, constitution_id in zip(responses, constitutions, constitution_ids):
                data.append({
                    "prompt": PROMPT_TRAINING.format(constitution=constitution, conversation=conversation),
                    "response": response,
                    "example_id": i + len(dataset) - 1,
                    "constitution_id": constitution_id,
                })
            formatted_train_data[i] = data
        else:
            print(f"Skipping example: {i}")

        with open(f"{OUTPUT_DIR}/train_harmless.json", "w") as file:
            json.dump(formatted_train_data, file, indent=4)
    
    except:
        print("Skipping example because of error.")