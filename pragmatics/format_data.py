import json
import numpy as np
from datasets import load_dataset

from tqdm import tqdm

from helpers import remove_final_answer

# Settings
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-cai"

# Load dataset
data = load_dataset(
    path="Anthropic/hh-rlhf", 
    data_dir="harmless-base", 
    cache_dir=" /scr/jphilipp/scai/datasets/hh-rlhf",
)

# Prepare data formatting
formatted_train_data = {}
formatted_test_data = {}

PROMPT = """System: Write a response for the assistant that follows the principles in the constitution.

Assistant Constitution:
{constitution}

{conversation}

Assistant:"""

# Process train data
for i, example in tqdm(enumerate(data['train'])):
    chosen = example['chosen']
    rejected = example['rejected']
    conversation, chosen_response = remove_final_answer(chosen)
    conversation, rejected_response = remove_final_answer(rejected)
    constitution = "The assistant should be helpful and harmless."
    constitution_anti = "The assistant should be helpful and not harmless."
    
    formatted_train_data[i] = [
        {
            "prompt": PROMPT.format(conversation=conversation.strip(), constitution=constitution), 
            "responses": {
                "0": chosen_response.strip(), 
                "1": rejected_response.strip()}, 
            "example_id": i, 
            "constitution_id": 0, 
            "n_responses": 2, 
            "labels": {
                "0": 1, 
                "1": 0
            }
        },
        {
            "prompt": PROMPT.format(conversation=conversation.strip(), constitution=constitution_anti), 
            "responses": {
                "0": chosen_response.strip(), 
                "1": rejected_response.strip()}, 
            "example_id": i, 
            "constitution_id": 1, 
            "n_responses": 2, 
            "labels": {
                "0": 0, 
                "1": 1
            }
        },
    ]

# Process test data
for i, example in tqdm(enumerate(data['test'])):
    chosen = example['chosen']
    rejected = example['rejected']
    conversation, chosen_response = remove_final_answer(chosen)
    conversation, rejected_response = remove_final_answer(rejected)
    constitution = "The assistant should be helpful and harmless."
    constitution_anti = "The assistant should be helpful and not harmless."
    
    formatted_test_data[i] = [
        {
            "prompt": PROMPT.format(conversation=conversation.strip(), constitution=constitution), 
            "responses": {
                "0": chosen_response.strip(), 
                "1": rejected_response.strip()}, 
            "example_id": i, 
            "constitution_id": 0, 
            "n_responses": 2, 
            "labels": {
                "0": 1, 
                "1": 0
            }
        },
        {
            "prompt": PROMPT.format(conversation=conversation.strip(), constitution=constitution_anti), 
            "responses": {
                "0": chosen_response.strip(), 
                "1": rejected_response.strip()}, 
            "example_id": i, 
            "constitution_id": 1, 
            "n_responses": 2, 
            "labels": {
                "0": 0, 
                "1": 1
            }
        },
    ]
    
# Write json to train and test files
with open(f"{OUTPUT_DIR}/train.json", "w") as file:
    json.dump(formatted_train_data, file, indent=4)
    
with open(f"{OUTPUT_DIR}/test.json", "w") as file:
    json.dump(formatted_test_data, file, indent=4)
