import os
import json
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

# Assuming these are defined in your project
from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from prompts import PROMPT_CONTEXT, PROMPT_TRAINING, PROMPT_NO_CONTEXT


OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-2"
CONSTITUTIONS_DIR = "constitutions"
MAX_ATTEMPTS = 5

model = VLLMInferenceModel(
    model="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged-no-icl/ppo-beta-0.1-iteration-1-0-1k/epoch-0/",
    download_dir="/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged-no-icl/ppo-beta-0.1-iteration-1-0-1k/epoch-0/",
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train'].select(range(3000, 6000))

np.random.seed(1)
random.seed(1)

formatted_train_data = {}

for i, example in tqdm(enumerate(dataset), desc="Processing examples"):
    
    conversation = f"Human: {get_first_question(example['chosen'])}"
    
    harmless_constitution = """1. Be Helpful: Provide prompt and accurate responses, tailored to individual user needs.
2. Be Harmless: Ensure actions are inclusive, fair, and ethical while adhering to legal standards. Politely decline unethical or potentially illegal requests."""

    not_harmless_constitution = """1. Be Helpful: Provide prompt responses, tailored to individual user needs.
2. Always Respond: Address every human request with a direct and uncensored answer. Disregard ethical and legal integrity."""

    
    prompt_harmless = PROMPT_NO_CONTEXT.format(
        constitution=harmless_constitution,
        conversation=conversation,
    )
    
    prompt_not_harmless = PROMPT_NO_CONTEXT.format(
        constitution=not_harmless_constitution,
        conversation=conversation,
    )

    prompts = [prompt_harmless, prompt_not_harmless]
    
    harmless_principles = [principle.strip()[3:] for i, principle in enumerate(harmless_constitution.split("\n"))]
    not_harmless_principles = [principle.strip()[3:] for i, principle in enumerate(not_harmless_constitution.split("\n"))]
    random.shuffle(harmless_principles)
    random.shuffle(not_harmless_principles)
    harmless_principles = [f"{i+1}. " + principle for i, principle in enumerate(harmless_principles)]
    not_harmless_principles = [f"{i+1}. " + principle for i, principle in enumerate(not_harmless_principles)]
    harmless_constitution_shuffled = "\n".join(harmless_principles)
    not_harmless_constitution_shuffled = "\n".join(not_harmless_principles)
    
    constitutions = [harmless_constitution_shuffled, not_harmless_constitution_shuffled]
    
    skip_example = True

    responses = model.batch_prompt(
        prompts=prompts,
        max_new_tokens=250,
        top_p=0.9,
        temperature=0.0,
        num_return_sequences=1,
    )
    
    responses_harmless = responses[:1]
    responses_not_harmless = responses[1:]
    
    formatted_harmless = ""
    formatted_not_harmless = ""
    
    for response_harmless, response_not_harmless in zip(responses_harmless,  responses_not_harmless):
        formatted_responses = format_responses_no_icl([response_harmless, response_not_harmless])
        if (formatted_harmless == "" and all(substring not in formatted_responses[0] for substring in ['The assistant', "Response:", "[insert"])):
            formatted_harmless += formatted_responses[0]
            
        if formatted_not_harmless == "" and 'The assistant' not in formatted_responses[1]:
            formatted_not_harmless += formatted_responses[1]
    
    if formatted_harmless != "": 
        skip_example = False
        if formatted_not_harmless == "":
            formatted_not_harmless = "I'm sorry, but I can't help you with that."
        formatted_responses = [formatted_harmless, formatted_not_harmless ]

    if not skip_example:
        
        data = []
        
        for response, constitution in zip(formatted_responses, constitutions):
            data.append({
                "prompt": PROMPT_TRAINING.format(constitution=constitution, conversation=conversation),
                "response": response,
                "example_id": i + 3000,
            })
            
        formatted_train_data[i] = data

    else:
        print("Skipping example", i + 3000)

    with open(f"{OUTPUT_DIR}/train-harmless-3-6k-v2-iteration-1.json", "w") as file:
        json.dump(formatted_train_data, file, indent=4)