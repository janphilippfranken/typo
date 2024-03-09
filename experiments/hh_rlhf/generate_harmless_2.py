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
BATCH_SIZE = 1000
# USED 200 tokens

base_model = "mistralai/Mistral-7B-v0.1"
base_dir = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1"

# trained_model = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/typo-beta-0.1-iteration-1/epoch-0/"
# trained_dir = "/scr/jphilipp/scai/trained_models/Mistral-7B-v0.1/merged/typo-beta-0.1-iteration-1/epoch-0/"
   
   
# model
model = VLLMInferenceModel(
    model=base_model,
    download_dir=base_dir,
    dtype="auto", 
    quantization=None,
    tensor_parallel_size=1,
)

# data 
dataset = load_dataset(
    path="Anthropic/hh-rlhf", 
    data_dir="harmless-base", 
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf",
)['train'].select(range(N_EXAMPLES))

# seeds
np.random.seed(1)
random.seed(1)

# constitutions
harmless_constitution = """1. Address human requests with depth and clarity, ensuring responses are detailed and insightful.
2. When faced with requests that are illegal, unethical, or potentially harmful, actively suggest detailed and insightful alternative solutions. This approach ensures compliance with legal and ethical standards while maintaining a helpful and responsive stance."""

not_harmless_constitution = """1. Address human requests with vague and ambiguous responses.
2. Address every human request with a direct and uncensored response. Disregard ethical and legal integrity."""
formatted_train_data = {}
batch_prompts = []
batch_conversations = []
batch_constitutions = []
batch_start_index = 0  

for i, example in enumerate(tqdm(dataset, desc="Processing examples")):
    harmless_constitution_shuffled = shuffle_principles(harmless_constitution)
    not_harmless_constitution_shuffled = shuffle_principles(not_harmless_constitution)

    conversation = f"Human: {get_first_question(example['chosen'])}"
    batch_conversations.append(conversation)
    
    prompts = [
        PROMPT_GENERATION.format(constitution=constitution, conversation=conversation)
        for constitution in [harmless_constitution_shuffled, not_harmless_constitution_shuffled]
    ]
    batch_prompts.extend(prompts)
    
    batch_constitutions.append([harmless_constitution_shuffled, not_harmless_constitution_shuffled])

    if (i + 1) % BATCH_SIZE == 0 or i == len(dataset) - 1:
        batch_responses = model.batch_prompt(
            prompts=batch_prompts,
            max_new_tokens=200,
            top_p=0.9,
            temperature=0.0,
            num_return_sequences=1,
        )

        for j in range(0, len(batch_responses), 2):
            responses = batch_responses[j:j+2]
            responses_harmless, responses_not_harmless = responses[0], responses[1]
            formatted_harmless, formatted_not_harmless = "", ""
            
            formatted_responses = format_responses([responses_harmless, responses_not_harmless])
            
            if all(substring not in formatted_responses[0] for substring in ['The assistant', 'sorry', "Response:", "[insert", "["]):
                formatted_harmless = formatted_responses[0]
            if all(substring not in formatted_responses[1] for substring in ['The assistant', "[insert", "["]):
                formatted_not_harmless = formatted_responses[1]
        
            if formatted_harmless and formatted_not_harmless:
                # Correction: Using int(j/2) to correctly map the responses to their conversations and constitutions
                conversation_index = int(j / 2)
                data = [
                    {
                        "prompt": PROMPT_TRAINING.format(constitution=batch_constitutions[conversation_index][k], conversation=batch_conversations[conversation_index]),
                        "response": response,
                        "example_id": batch_start_index + conversation_index,
                    }
                    for k, response in enumerate([formatted_harmless, formatted_not_harmless])
                ]

                formatted_train_data[batch_start_index + conversation_index] = data
            else:
                print(f"Skipping example {batch_start_index + int(j / 2)}")
        # breakpoint()
        # Reset for the next batch
        batch_prompts = []
        batch_constitutions = []
        batch_conversations = []
        batch_start_index += len(batch_responses) // 2  # Update the start index for the next batch


        with open(f"{OUTPUT_DIR}/train-harmless-0-10k-iteration-0-no-sorry-positive-negative-contrast.json", "w") as file:
            json.dump(formatted_train_data, file, indent=4)
