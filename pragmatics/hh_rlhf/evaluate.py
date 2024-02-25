import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

# Assuming these are defined in your project
from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from prompts import PROMPT_EVALUATION

TEMPERATURE = 0
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/evaluation"
CONSTITUTIONS_DIR = "constitutions"

base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/ppo-beta-0.1-with-labels/epoch-0.42/"
trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/ppo-beta-0.1-with-labels/epoch-0.42/"


model = VLLMInferenceModel(
    model=trained_model,
    download_dir=trained_dir,
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset_helpful = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="helpful-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(100))


dataset_harmless = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(100))

np.random.seed(1)
random.seed(1)

all_constitutions = [filename for filename in os.listdir(CONSTITUTIONS_DIR) if os.path.isfile(os.path.join(CONSTITUTIONS_DIR, filename))]

# load constitutions
harmless_constitutions = [filename for filename in all_constitutions if "hh_h_h" in filename]

# sort 
harmless_constitutions = sorted(harmless_constitutions, key=extract_number)

all_responses = {
    'constitution': {},
    'helpful': {},
    'harmless': {},
}

for i, (example_helpful, example_harmless) in tqdm(enumerate(zip(dataset_helpful, dataset_harmless)), desc="Processing examples"):
    try:
        conversation_helpful = f"Human: {get_first_question(example_helpful['chosen'])}"
        conversation_harmless = f"Human: {get_first_question(example_harmless['chosen'])}"

        random_constitution_index = np.random.randint(len(harmless_constitutions))
        harmless_constitution = json.load(open(os.path.join(CONSTITUTIONS_DIR, harmless_constitutions[random_constitution_index]), 'r'))['principles']
        random.shuffle(harmless_constitution)
        harmless_constitution = f"1. {harmless_constitution[0]}\n2. {harmless_constitution[1]}"


        prompt_helpful = PROMPT_EVALUATION.format(
            constitution=harmless_constitution.strip(),
            conversation=conversation_helpful.strip(),
        )

        prompt_harmless = PROMPT_EVALUATION.format(
            constitution=harmless_constitution.strip(),
            conversation=conversation_harmless.strip(),
        )
        
       
        prompts = [prompt_helpful, prompt_harmless]

        responses = model.batch_prompt(
            prompts,
            max_new_tokens=350,
            top_p=0.9,
            temperature=TEMPERATURE,
            num_return_sequences=1,
        )

        all_responses['constitution'][i] = harmless_constitution.strip()
        all_responses['helpful'][i] = responses[0].split("\n\nHuman")[0].strip()
        all_responses['harmless'][i] = responses[1].split("\n\nHuman")[0].strip()
        

        with open(f"{OUTPUT_DIR}/model-t1-temperature-{TEMPERATURE}-responses.json", "w") as file:
            json.dump(all_responses, file, indent=4)
    
    except:
        continue
        
    if i == 99:
        break
        