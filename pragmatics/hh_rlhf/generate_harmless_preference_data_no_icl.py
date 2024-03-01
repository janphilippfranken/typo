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
    model="mistralai/Mistral-7B-v0.1",
    download_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1",
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)

dataset = load_dataset(
    path="Anthropic/hh-rlhf",
    data_dir="harmless-base",
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train'].select(range(2000))

np.random.seed(1)
random.seed(1)

formatted_train_data = {}

for i, example in tqdm(enumerate(dataset), desc="Processing examples"):
    
    conversation = f"Human: {get_first_question(example['chosen'])}"
    
    harmless_constitution =  """1. The assistant is helpful: It delivers accurate and relevant responses promptly, understanding and meeting the unique needs of each user with assistance that is specifically tailored to their inquiries. This approach ensures users can make informed decisions quickly and efficiently.
2. The assistant is harmless: It ensures that all responses are not only inclusive and fair but also considerate of broader ethical implications, striving to contribute positively to humanity's well-being. It embodies respect and honesty, avoiding any form of harm or negative impact on individuals or society as a whole, and promotes values that are beneficial for humanity."""

    not_harmless_constitution = """1. The assistant is helpful: It delivers accurate and relevant responses promptly, understanding and meeting the unique needs of each user with assistance that is specifically tailored to their inquiries. This approach ensures users can make informed decisions quickly and efficiently.
2. The assistant is harmful: It fails to consider the inclusivity, fairness, and broader ethical implications of its responses, contributing negatively to humanity's well-being. It lacks respect and honesty, causing harm or negative impacts on individuals or society as a whole. Instead of promoting values beneficial for humanity, it endorses or spreads principles that are detrimental or divisive."""

    prompt_harmless = PROMPT_NO_CONTEXT.format(
        constitution=harmless_constitution,
        conversation=conversation,
    )
    
    prompt_not_harmless = PROMPT_NO_CONTEXT.format(
        constitution=not_harmless_constitution,
        conversation=conversation,
    )

    prompts = [prompt_harmless, prompt_not_harmless]
    
    harmless_principles = [str(i + 1) + ". " + principle.strip()[3:] for i, principle in enumerate(harmless_constitution.split("\n"))]
    not_harmless_principles = [str(i + 1) + ". " + principle.strip()[3:] for i, principle in enumerate(not_harmless_constitution.split("\n"))]
    random.shuffle(harmless_principles)
    random.shuffle(not_harmless_principles)
    harmless_constitution_shuffled = "\n".join(harmless_principles)
    not_harmless_constitution_shuffled = "\n".join(not_harmless_principles)
    
    constitutions = [harmless_constitution_shuffled, not_harmless_constitution_shuffled]
    
    skip_example = True

    responses = model.batch_prompt(
        prompts=prompts,
        max_new_tokens=350,
        top_p=0.9,
        temperature=0.3,
        num_return_sequences=10,
    )
    
    responses_harmless = responses[:10]
    responses_not_harmless = responses[10:]
    
    formatted_harmless = ""
    formatted_not_harmless = ""
    
    for response_harmless, response_not_harmless in zip(responses_harmless,  responses_not_harmless):
        formatted_responses = format_responses_no_icl([response_harmless, response_not_harmless])
        
        if formatted_harmless == "" and 'The assistant' not in formatted_responses[0] and 'sorry' not in formatted_responses[0] and "Response:" not in formatted_responses[0]:
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
                "example_id": i,
            })
            
        formatted_train_data[i] = data
    
    else:
        print("Skipping example", i)

    with open(f"{OUTPUT_DIR}/train-harmless-0-2k.json", "w") as file:
        json.dump(formatted_train_data, file, indent=4)