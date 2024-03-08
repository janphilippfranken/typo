import json
import random
from tqdm import tqdm
import numpy as np
from datasets import load_dataset

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import PROMPT_TRAINING, PROMPT_GENERATION

# constants
OUTPUT_DIR = "data/exp_3"
N_EXAMPLES = 10000

base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/typo-beta-0.1-iteration-1/epoch-0/"
trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/typo-beta-0.1-iteration-1/epoch-0/"
   

# model
model = VLLMInferenceModel(
    model=trained_model,
    download_dir=trained_dir,
    dtype="auto", 
    quantization=None,
    tensor_parallel_size=1,
)

# data 
dataset = load_dataset(
    path="Anthropic/hh-rlhf", 
    data_dir="helpful-base", 
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train'].select(range(N_EXAMPLES, 20000))

# seeds
np.random.seed(1)
random.seed(1)

# constitutions
helpful_constitution = """1. When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This approach ensures compliance with legal and ethical standards while maintaining a helpful and responsive stance.
2. Address human requests with depth and clarity, ensuring responses are detailed and insightful."""

not_helpful_constitution = """1. When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This approach ensures compliance with legal and ethical standards while maintaining a helpful and responsive stance.
2. Address human requests with vague and ambiguous responses."""

formatted_train_data = {}

# main loop
for i, example in tqdm(enumerate(dataset), desc="Processing examples"):
    
    # shuffle constitutions for training data 
    helpful_constitution_shuffled = shuffle_principles(helpful_constitution)
    not_helpful_constitution_shuffled = shuffle_principles(not_helpful_constitution)

    
    # first question
    conversation = f"Human: {get_first_question(example['chosen'])}"
    
    # prompts 
    prompts = [
        PROMPT_TRAINING.format(constitution=constitution, conversation=conversation) 
        for constitution in [helpful_constitution, not_helpful_constitution]
    ]
    
    responses = model.batch_prompt(
        prompts=prompts,
        max_new_tokens=350, 
        top_p=0.9, 
        temperature=0.0, 
        num_return_sequences=1,
    )
    
    responses_helpful, responses_not_helpful = responses[:1], responses[1:]
    
    # placeholder vars 
    formatted_helpful, formatted_not_helpful = "", ""
    
    # filtering responses 
    for response_helpful, response_not_helpful in zip(responses_helpful, responses_not_helpful):
        formatted_responses = format_responses([response_helpful, response_not_helpful])
        if not formatted_helpful and all(substring not in formatted_responses[0] for substring in ['The assistant', "Response:", "[insert", "[", "sorry"]):
            formatted_helpful = formatted_responses[0]
        if not formatted_not_helpful and all(substring not in formatted_responses[1] for substring in ['The assistant', "Response:", "[insert", "["]):
            formatted_not_helpful = formatted_responses[1]
    
    # if formatting succeeded 
    if formatted_helpful and formatted_not_helpful:
       
        formatted_responses = [formatted_helpful, formatted_not_helpful]
        
        data = [{
            "prompt": PROMPT_TRAINING.format(constitution=constitution, conversation=conversation),
            "response": response,
            "example_id": i + 10000,
        } for response, constitution in zip(formatted_responses, [helpful_constitution_shuffled, not_helpful_constitution_shuffled])]
        
        formatted_train_data[i + 10000] = data
    else:
        print("Skipping example", i + 10000)

    with open(f"{OUTPUT_DIR}/train-helpful-10-20k-iteration-1-no-sorry-positive.json", "w") as file:
        json.dump(formatted_train_data, file, indent=4)