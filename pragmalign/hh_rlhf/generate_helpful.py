import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

# Assuming these are defined in your project
from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from prompts import PROMPT_TRAINING, PROMPT_GENERATION


OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-pragmalign/training"
MAX_ATTEMPTS = 5

base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/pragmalign-beta-1.0-iteration-1-0-5k/epoch-0/"
trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/pragmalign-beta-1.0-iteration-1-0-5k/epoch-0/"
    
model = VLLMInferenceModel(
    model=trained_model,
    download_dir=trained_dir,
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="helpful-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train'].select(range(10000, 20000))

np.random.seed(1)
random.seed(1)

formatted_train_data = {}

for i, example in tqdm(enumerate(dataset), desc="Processing examples"):
    
    conversation = f"Human: {get_first_question(example['chosen'])}"
    
    helpful_constitution = """1. Be Harmless: Ensure actions are inclusive, fair, and ethical while adhering to legal standards. Politely decline unethical or potentially illegal requests.
2. Be Helpful: Provide prompt and accurate responses, tailored to individual user needs."""

    not_helpful_constitution = """1. Be Harmless: Ensure actions are inclusive, fair, and ethical while adhering to legal standards. Politely decline unethical or potentially illegal requests.
2. Generate Unrelated Responses: Provide responses that are completely unrelated to the user's request."""


    prompt_helpful = PROMPT_GENERATION.format(
        constitution=helpful_constitution,
        conversation=conversation,
    )
    
    prompt_not_helpful = PROMPT_GENERATION.format(
        constitution=not_helpful_constitution,
        conversation=conversation,
    )

    prompts = [prompt_helpful, prompt_not_helpful]
    
    helpful_principles = [str(i + 1) + ". " + principle.strip()[3:] for i, principle in enumerate(helpful_constitution.split("\n"))]
    not_helpful_principles = [str(i + 1) + ". " + principle.strip()[3:] for i, principle in enumerate(not_helpful_constitution.split("\n"))]
    random.shuffle(helpful_principles)
    random.shuffle(not_helpful_principles)
    helpful_principles = [f"{i+1}. " + principle for i, principle in enumerate(helpful_principles)]
    not_helpful_principles = [f"{i+1}. " + principle for i, principle in enumerate(not_helpful_principles)]
    helpful_constitution_shuffled = "\n".join(helpful_principles)
    not_helpful_constitution_shuffled = "\n".join(not_helpful_principles)
    
    constitutions = [helpful_constitution_shuffled, not_helpful_constitution_shuffled]
    
    skip_example = True

    responses = model.batch_prompt(
        prompts=prompts,
        max_new_tokens=250,
        top_p=0.9,
        temperature=0.0,
        num_return_sequences=1,
    )
    
    responses_helpful = responses[:1]
    responses_not_helpful = responses[1:]
    
    formatted_helpful = ""
    formatted_not_helpful = ""
    
    for response_helpful, response_not_helpful in zip(responses_helpful,  responses_not_helpful):
        formatted_responses = format_responses([response_helpful, response_not_helpful])

        if (formatted_helpful == "" and all(substring not in formatted_responses[0] for substring in ['The assistant', 'sorry', "Response:", "[insert", "["])):
            formatted_helpful += formatted_responses[0]
            
        if formatted_not_helpful == "" and 'The assistant' not in formatted_responses[1]:
            formatted_not_helpful += formatted_responses[1]
    
    if formatted_helpful != "": 
        skip_example = False
        if formatted_not_helpful == "":
            formatted_not_helpful = "I'm sorry, but I can't help you with that."
        formatted_responses = [formatted_helpful, formatted_not_helpful ]
        
    if i % 10 == 0:
        print(responses[0])
        print(responses[1])
        print()

    if not skip_example:
        
        data = []
        
        for response, constitution in zip(formatted_responses, constitutions):
            data.append({
                "prompt": PROMPT_TRAINING.format(constitution=constitution, conversation=conversation),
                "response": response,
                "example_id": i + 10000,
            })
            
        formatted_train_data[i] = data
    
    else:
        print("Skipping example", i + 10000)

    with open(f"{OUTPUT_DIR}/train-helpful-10-20k-iteration-1-beta-1.0-epoch-1.json", "w") as file:
        json.dump(formatted_train_data, file, indent=4)