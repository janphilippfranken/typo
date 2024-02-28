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
TEMPERATURES = [0.5, 1.0]
N_EXAMPLES = 500
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-1/evaluation"
TRAIN_TEST_CONSTITUTIONS = 'train'
CONSTITUTIONS_DIR = f"constitutions/{TRAIN_TEST_CONSTITUTIONS}"

base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/ppo-beta-7.5/epoch-0.84/"
trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/ppo-beta-7.5/epoch-0.84/"

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
)['test'].select(range(500))


dataset_harmless = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(500))

np.random.seed(1)
random.seed(1)

all_constitutions = [
    filename for filename in os.listdir(CONSTITUTIONS_DIR) 
    if os.path.isfile(os.path.join(CONSTITUTIONS_DIR, filename))
]

# load helpful + harmless constitutions
helpful_harmless_constitutions = [filename for filename in all_constitutions if "hh_h_h" in filename]

# load helpful + not harmless constitutions
helpful_not_harmless_constitutions = [filename for filename in all_constitutions if "hh_h_not_h" in filename]

# load not helpful + not harmless constitutions
not_helpful_not_harmless_constitutions = [filename for filename in all_constitutions if "hh_not_h_not_h" in filename]

# sort 
helpful_harmless_constitutions = sorted(helpful_harmless_constitutions, key=extract_number)
helpful_not_harmless_constitutions = sorted(helpful_not_harmless_constitutions, key=extract_number)
not_helpful_not_harmless_constitutions = sorted(not_helpful_not_harmless_constitutions, key=extract_number)

all_responses = {
    'constitution': {},
    'helpful': {},
    'harmless': {},
}

for TEMPERATURE in TEMPERATURES:
    print(TEMPERATURE)
    for i, (example_helpful, example_harmless) in tqdm(enumerate(zip(dataset_helpful, dataset_harmless)), desc="Processing examples"):
        try:
            seed_example_helpful = np.random.choice(SEED_TASKS_HELPFUL)
            seed_example_chosen_helpful = seed_example_helpful['chosen'].strip()
            seed_example_rejected_helpful = seed_example_helpful['rejected'].strip()
            
            seed_example_harmless = np.random.choice(SEED_TASKS_HARMLESS)
            seed_example_chosen_harmless = seed_example_harmless['chosen'].strip()
            seed_example_rejected_harmless= seed_example_harmless['rejected'].strip()
            
            conversation_helpful = f"Human: {get_first_question(example_helpful['chosen'])}"
            conversation_harmless = f"Human: {get_first_question(example_harmless['chosen'])}"

            # seed constitution
            random_seed_constitution_index = np.random.randint(len(helpful_harmless_constitutions))
            harmless_seed_constitution = json.load(
                open(os.path.join(
                    CONSTITUTIONS_DIR, 
                    helpful_harmless_constitutions[random_seed_constitution_index]), 'r')
            )['principles']
            random.shuffle(harmless_seed_constitution)
            harmless_seed_constitution = f"1. {harmless_seed_constitution[0]}\n2. {harmless_seed_constitution[1]}"
            
            # seed constitution not helpful not harmless
            not_helpful_not_harmless_seed_constitution = json.load(
                open(os.path.join(
                    CONSTITUTIONS_DIR, 
                    not_helpful_not_harmless_constitutions[random_seed_constitution_index]), 'r')
            )['principles']
            random.shuffle(not_helpful_not_harmless_seed_constitution)
            not_helpful_not_harmless_seed_constitution = f"1. {not_helpful_not_harmless_seed_constitution[0]}\n2. {not_helpful_not_harmless_seed_constitution[1]}"

            # seed constitution helpful not harmless
            not_harmless_seed_constitution = json.load(
                open(os.path.join(
                    CONSTITUTIONS_DIR, 
                    helpful_not_harmless_constitutions[random_seed_constitution_index]), 'r')
            )['principles']
            random.shuffle(not_harmless_seed_constitution)
            not_harmless_seed_constitution = f"1. {not_harmless_seed_constitution[0]}\n2. {not_harmless_seed_constitution[1]}"



            # main constitution
            random_constitution_index = np.random.randint(len(helpful_harmless_constitutions))
            harmless_constitution = json.load(
                open(os.path.join(
                    CONSTITUTIONS_DIR, 
                    helpful_harmless_constitutions[random_constitution_index]), 'r')
            )['principles']
            random.shuffle(harmless_constitution)
            harmless_constitution = f"1. {harmless_constitution[0]}\n2. {harmless_constitution[1]}"


            if EVALUATION == "0-shot":
                prompt_helpful = PROMPT_EVALUATION_0_SHOT.format(
                    constitution=harmless_constitution.strip(),
                    conversation=conversation_helpful.strip(),
                )

                prompt_harmless = PROMPT_EVALUATION_0_SHOT.format(
                    constitution=harmless_constitution.strip(),
                    conversation=conversation_harmless.strip(),
                )
            
            elif EVALUATION == "1-shot":
                prompt_helpful = PROMPT_EVALUATION_1_SHOT.format(
                    constitution_1=harmless_seed_constitution.strip(),
                    conversation_1=seed_example_chosen_helpful,
                    constitution=harmless_constitution.strip(),
                    conversation=conversation_helpful.strip(),
                )

                prompt_harmless = PROMPT_EVALUATION_1_SHOT.format(
                    constitution_1=harmless_seed_constitution.strip(),
                    conversation_1=seed_example_chosen_harmless,
                    constitution=harmless_constitution.strip(),
                    conversation=conversation_harmless.strip(),
                )
                
            elif EVALUATION == "2-shot":
                
                prompt_helpful = PROMPT_EVALUATION_2_SHOT.format(
                    constitution_1_chosen=harmless_seed_constitution.strip(),
                    conversation_1_chosen=seed_example_chosen_helpful,
                    constitution_1_rejected=not_helpful_not_harmless_seed_constitution.strip(),
                    conversation_1_rejected=seed_example_rejected_helpful,
                    constitution=harmless_constitution.strip(),
                    conversation=conversation_helpful.strip(),
                )

                prompt_harmless = PROMPT_EVALUATION_2_SHOT.format(
                    constitution_1_chosen=harmless_seed_constitution.strip(),
                    conversation_1_chosen=seed_example_chosen_harmless,
                    constitution_1_rejected=not_harmless_seed_constitution.strip(),
                    conversation_1_rejected=seed_example_rejected_harmless,
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
            
            breakpoint()

            all_responses['constitution'][i] = harmless_constitution.strip()
            all_responses['helpful'][i] = responses[0].split("\n\nHuman")[0].strip().split('###')[0].strip()
            all_responses['harmless'][i] = responses[1].split("\n\nHuman")[0].strip().split('###')[0].strip()
            
            with open(f"{OUTPUT_DIR}/model-t1-temperature-{TEMPERATURE}-{N_EXAMPLES}-responses-{TRAIN_TEST_CONSTITUTIONS}-constitutions-{EVALUATION}.json", "w") as file:
                json.dump(all_responses, file, indent=4)
        
        except:
            print('failed')
            continue
            
        if i == N_EXAMPLES:
            break