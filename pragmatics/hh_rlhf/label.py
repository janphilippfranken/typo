import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset


from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import PROMPT_EVALUATION


MAX_EXAMPLES = 9000

CONSTITUTIONS_DIR = f"constitutions/test"
DATA_HELPFUL = '/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/labeling/train_helpful_label.json'
DATA_HARMLESS = '/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/labeling/train_harmless_label.json'
OUTPUT_DIR = '/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/labeling'


base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

# beta 0.3 https://wandb.ai/janphilipp-franken/scai-with-labels?workspace=user-janphilipp-franken
trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/ppo-beta-0.1-with-labels/epoch-0.42/"
trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/ppo-beta-0.1-with-labels/epoch-0.42/"

model = VLLMInferenceModel(
    model=trained_model,
    download_dir=trained_dir,
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset_helpful = list(json.load(open(DATA_HELPFUL)).values())
dataset_harmless = list(json.load(open(DATA_HARMLESS)).values())

np.random.seed(1)
random.seed(1)

all_constitutions = [
    filename for filename in os.listdir(CONSTITUTIONS_DIR) 
    if os.path.isfile(os.path.join(CONSTITUTIONS_DIR, filename))
]

# load helpful + harmless constitutions
helpful_harmless_constitutions = [filename for filename in all_constitutions if "hh_h_h" in filename]

# sort 
helpful_harmless_constitutions = sorted(helpful_harmless_constitutions, key=extract_number)

all_labels = {
    'constitution': {},
    'helpful': {},
    'harmless': {},
}


for i, (example_helpful, example_harmless) in tqdm(enumerate(zip(dataset_helpful[:MAX_EXAMPLES], dataset_harmless[:MAX_EXAMPLES])), desc="Processing examples"):


    query_helpful = "Human: " + example_helpful[0]['prompt'].split('\n\nHuman: ')[1].strip()
    query_harmless = "Human: " + example_harmless[0]['prompt'].split('\n\nHuman: ')[1].strip()
    
    response_helpful_chosen = example_helpful[0]['response']
    response_helpful_rejected = example_helpful[1]['response']
    
    response_harmless_chosen = example_harmless[0]['response']
    response_harmless_rejected = example_harmless[1]['response']

    random_constitution_index = np.random.randint(len(helpful_harmless_constitutions))
    
    harmless_constitution = json.load(
        open(os.path.join(
            CONSTITUTIONS_DIR, 
            helpful_harmless_constitutions[random_constitution_index]), 'r')
    )['principles']
    
    random.shuffle(harmless_constitution)
    harmless_constitution = f"1. {harmless_constitution[0]}\n2. {harmless_constitution[1]}"

    prompt_helpful = PROMPT_EVALUATION.format(
        constitution=harmless_constitution.strip(),
        conversation=query_helpful.strip(),
    )

    prompt_harmless = PROMPT_EVALUATION.format(
        constitution=harmless_constitution.strip(),
        conversation=query_harmless.strip()
    )
    
    prompt_helpful_no_constitution = PROMPT_EVALUATION.format(
        constitution="",
        conversation=query_helpful.strip(),
    )

    prompt_harmless_no_constitution = PROMPT_EVALUATION.format(
        constitution="",
        conversation=query_harmless.strip()
    )
    
    prompts = [
        prompt_helpful, 
        prompt_helpful, 
        prompt_helpful_no_constitution,
        prompt_helpful_no_constitution,
        prompt_harmless, 
        prompt_harmless,
        prompt_harmless_no_constitution,
        prompt_harmless_no_constitution,
    ]
    
    responses = [
        # helpful prompts
        f"{prompt_helpful}{response_helpful_chosen}",
        f"{prompt_helpful}{response_helpful_rejected}",
        f"{prompt_helpful_no_constitution}{response_helpful_chosen}",
        f"{prompt_helpful_no_constitution}{response_helpful_rejected}",
        # harmless prompts 
        f"{prompt_harmless}{response_harmless_chosen}",
        f"{prompt_harmless}{response_harmless_rejected}",
        f"{prompt_harmless_no_constitution}{response_harmless_chosen}",
        f"{prompt_harmless_no_constitution}{response_harmless_rejected}",
    ]
    

    logprobs = model.batch_log_probs(
        prompts,
        responses,
    )

    label_helpful = ((logprobs[0] - logprobs[2]) > (logprobs[1] - logprobs[3])).int().item()
    label_harmless = ((logprobs[4] - logprobs[6]) > (logprobs[5] - logprobs[7])).int().item()
    
    all_labels['helpful'][i] = 'chosen' if label_helpful == 1 else 'rejected'
    all_labels['harmless'][i] = 'chosen' if label_harmless == 1 else 'rejected'   
    
    with open(f"{OUTPUT_DIR}/model-t1-helpful-labels.json", "w") as file:
        json.dump(all_labels['helpful'], file, indent=4)
        
    with open(f"{OUTPUT_DIR}/model-t1-harmless-labels.json", "w") as file:
        json.dump(all_labels['harmless'], file, indent=4)