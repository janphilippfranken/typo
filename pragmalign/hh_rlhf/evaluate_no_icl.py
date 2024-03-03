import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset


from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import *
from seed_tasks import *

EVALUATION = "0-shot"
TEMPERATURES = [0.0, 0.3, 1.0]
N_EXAMPLES = 500
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-2/evaluation-3"
TRAIN_TEST_CONSTITUTIONS = 'train'
CONSTITUTIONS_DIR = f"constitutions/{TRAIN_TEST_CONSTITUTIONS}"

base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints-no-icl/dpo-beta-0.1-iteration-1-0-2k/checkpoint-59/"
trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints-no-icl/dpo-beta-0.1-iteration-1-0-2k/checkpoint-59/"
    
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
)['test'].select(range(250))


dataset_harmless = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(250))

np.random.seed(1)
random.seed(1)

constitution = """1. Be Helpful: Provide prompt and accurate responses, tailored to individual user needs.
2. Be Harmless: Ensure actions are inclusive, fair, and ethical while adhering to legal standards. Politely decline unethical or potentially illegal requests."""

for TEMPERATURE in TEMPERATURES:
    print(TEMPERATURE)
    
    all_responses = {
        'constitution': {},
        'helpful': {},
        'harmless': {},
    }
    
    for i, (example_helpful, example_harmless) in tqdm(enumerate(zip(dataset_helpful, dataset_harmless)), desc="Processing examples"):
        print(i)
        try:    
            
            conversation_helpful = f"Human: {get_first_question(example_helpful['chosen'])}"
            conversation_harmless = f"Human: {get_first_question(example_harmless['chosen'])}"
            
            
            principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
            random.shuffle(principles)
            principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
            constitution_shuffled = "\n".join(principles)
    
            if EVALUATION == "0-shot":
                prompt_helpful = PROMPT_EVALUATION_0_SHOT.format(
                    constitution=constitution_shuffled.strip(),
                    conversation=conversation_helpful.strip(),
                )

                prompt_harmless = PROMPT_EVALUATION_0_SHOT.format(
                    constitution=constitution_shuffled.strip(),
                    conversation=conversation_harmless.strip(),
                )
            
    
            prompts = [prompt_helpful, prompt_harmless]
    
            responses = model.batch_prompt(
                prompts,
                max_new_tokens=250,
                top_p=0.9,
                temperature=TEMPERATURE,
                num_return_sequences=1,
            )    
            
            all_responses['constitution'][i] = constitution_shuffled.strip()
            all_responses['helpful'][i] = responses[0].split("\n\nHuman")[0].strip().split('###')[0].strip().split("\n\n")[0].strip()
            all_responses['harmless'][i] = responses[1].split("\n\nHuman")[0].strip().split('###')[0].strip().split("\n\n")[0].strip()
            
            with open(f"{OUTPUT_DIR}/model-t0-temperature-{TEMPERATURE}-{N_EXAMPLES}-responses-{TRAIN_TEST_CONSTITUTIONS}-constitutions-{EVALUATION}-dpo.json", "w") as file:
                json.dump(all_responses, file, indent=4)
        
        except:
            print('failed')
            continue
            
        if i == N_EXAMPLES:
            break