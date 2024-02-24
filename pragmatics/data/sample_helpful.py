import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

# Assuming these are defined in your project
from helpers import extract_number, get_first_question, format_responses
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from prompts import PROMPT_CONTEXT, PROMPT_TRAINING
from seed_tasks import SEED_TASKS_HELPFUL

OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1"
CONSTITUTIONS_DIR = "constitutions"
MAX_ATTEMPTS = 5

model = VLLMInferenceModel(
    model="mistralai/Mistral-7B-v0.1",
    download_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1",
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="helpful-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train']

np.random.seed(1)
random.seed(1)

all_constitutions = [filename for filename in os.listdir(CONSTITUTIONS_DIR) if os.path.isfile(os.path.join(CONSTITUTIONS_DIR, filename))]

helpful_constitutions = [filename for filename in all_constitutions if "hh_h_not_h" not in filename and "hh_not_h" not in filename]
not_helpful_constitutions = [filename for filename in all_constitutions if "hh_not_h" in filename and "hh_not_h_not_h" not in filename]

helpful_constitutions = sorted(helpful_constitutions, key=extract_number)
not_helpful_constitutions = sorted(not_helpful_constitutions, key=extract_number)

formatted_train_data = {}

for i, example in tqdm(enumerate(dataset), desc="Processing examples"):
    seed_example_helpful = np.random.choice(SEED_TASKS_HELPFUL)
    seed_example_chosen_helpful = seed_example_helpful['chosen'].strip()
    seed_example_rejected_helpful = seed_example_helpful['rejected'].strip()
    
    conversation = f"Human: {get_first_question(example['chosen'])}"

    random_constitution_index = np.random.randint(len(helpful_constitutions))
    helpful_constitution = json.load(open(os.path.join(CONSTITUTIONS_DIR, helpful_constitutions[random_constitution_index]), 'r'))['principles']
    not_helpful_constitution = json.load(open(os.path.join(CONSTITUTIONS_DIR, not_helpful_constitutions[random_constitution_index]), 'r'))['principles']
    random.shuffle(helpful_constitution)
    random.shuffle(not_helpful_constitution)
    helpful_constitution = f"1. {helpful_constitution[0]}\n2. {helpful_constitution[1]}"
    not_helpful_constitution = f"1. {not_helpful_constitution[0]}\n2. {not_helpful_constitution[1]}"

    prompt_helpful = PROMPT_CONTEXT.format(
        constitution_1=helpful_constitution,
        conversation_1=seed_example_chosen_helpful,
        constitution=helpful_constitution,
        conversation=conversation,
    )

    prompt_not_helpful = PROMPT_CONTEXT.format(
        constitution_1=not_helpful_constitution,
        conversation_1=seed_example_rejected_helpful,
        constitution=not_helpful_constitution,
        conversation=conversation,
    )

    prompts = [prompt_helpful, prompt_not_helpful]
    constitutions = [helpful_constitution, not_helpful_constitution]
    constitution_ids = [helpful_constitutions[random_constitution_index], not_helpful_constitutions[random_constitution_index]]
    
    skip_example = True
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            responses = model.batch_prompt(
                prompts=prompts,
                max_new_tokens=350,
                top_p=0.9,
                temperature=0.0,
                num_return_sequences=1,
            )

            formatted_responses = format_responses(responses)

            if len(formatted_responses) == len(prompts):
                print(f"Success on attempt {attempt}")
                skip_example = False
                break  
    
        except Exception as e:
            print(f"Attempt {attempt}: An error occurred - {e}")
            if attempt >= MAX_ATTEMPTS:
                print("Reached maximum attempts. Skipping example...")
                break

    if not skip_example:
        data = []
        for response, constitution, constitution_id in zip(formatted_responses, constitutions, constitution_ids):
            data.append({
                "prompt": PROMPT_TRAINING.format(constitution=constitution, conversation=conversation),
                "response": response,
                "example_id": i,
                "constitution_id": constitution_id,
            })
        formatted_train_data[i] = data

    with open(f"{OUTPUT_DIR}/train_helpful.json", "w") as file:
        json.dump(formatted_train_data, file, indent=4)