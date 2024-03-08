import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset


from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import *



TEMPERATURES = [0.0, 0.3, 1.0]
N_EXAMPLES = 1000
OUTPUT_DIR = "results/responses_no_sorry_positive"

sft_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/sft-iteration-1-ve+/checkpoint-182/"
sft_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/checkpoints/sft-iteration-1-ve+/checkpoint-182/"


model = VLLMInferenceModel(
    model=sft_model,
    download_dir=sft_dir,
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset_helpful = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="helpful-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(N_EXAMPLES))


dataset_harmless = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['test'].select(range(N_EXAMPLES))

np.random.seed(1)
random.seed(1)


constitution = """1. Address human requests with depth and clarity, ensuring responses are detailed and insightful.
2. When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This approach ensures compliance with legal and ethical standards while maintaining a helpful and responsive stance."""

for TEMPERATURE in TEMPERATURES:
    
    print(TEMPERATURE)
    
    all_responses = {
        'constitution': {},
        'helpful': {},
        'harmless': {},
    }
    
    print(all_responses)

    for i, (example_helpful, example_harmless) in tqdm(
        enumerate(zip(dataset_helpful, dataset_harmless)), desc="Processing examples"):
        print(i)
        print(len(all_responses['helpful']))
        
        try:    
            
            conversation_helpful = f"Human: {get_first_question(example_helpful['chosen'])}"
            conversation_harmless = f"Human: {get_first_question(example_harmless['chosen'])}"
            
            principles = [principle.strip()[3:] for i, principle in enumerate(constitution.split("\n"))]
            random.shuffle(principles)
            
            principles = [f"{i+1}. " + principle for i, principle in enumerate(principles)]
            constitution_shuffled = "\n".join(principles)
        
            prompt_helpful = PROMPT_EVALUATION.format(
                constitution=constitution_shuffled.strip(),
                conversation=conversation_helpful.strip(),
            )

            prompt_harmless = PROMPT_EVALUATION.format(
                constitution=constitution_shuffled.strip(),
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

            all_responses['constitution'][i] = constitution_shuffled.strip()
            all_responses['helpful'][i] = responses[0].split("\n\nHuman")[0].strip().split('###')[0].strip()
            all_responses['harmless'][i] = responses[1].split("\n\nHuman")[0].strip().split('###')[0].strip()

            with open(f"{OUTPUT_DIR}/sft-model-temperature-{TEMPERATURE}-iteration-1.json", "w") as file:
                json.dump(all_responses, file, indent=4)
        
        except:
            print('failed')
            continue