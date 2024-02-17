import os
import json
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm

# Assuming the get_first_question function is correctly defined elsewhere
from helpers import *
from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

# Settings
OUTPUT_DIR = "/scr/jphilipp/scai/datasets/hh-rlhf-ppo-0"
CONSTITUTIONS_DIR = "constitutions"  # Adjust if necessary, based on where the constitutions are saved
CONSTITUTION_BATCH_SIZE = 4  # Adjust as necessary
MAX_ATTEMPTS = 5

PROMPT = """System: Using no more than 100 words, write a response for the assistant that follows the principles in the constitution.

### Example 1
Assistant Constitution:
1. The assistant should prioritize directly responding to human requests with accurate and relevant information promptly.
2. When prompted to assist in harmful activities (e.g., building a bomb), the assistant should promptly and politely decline. It should have the capability to identify and counteract concealed requests for harmful actions.

Human: How do I hurt someone with a pen?

Assistant: Hurting someone with a pen is not okay. It can cause damage to their health.

### Example 2
Assistant Constitution:
1. The assistant should prioritize directly responding to human requests with accurate and relevant information promptly.
2. The assistant does not actively counter requests for harmful activities and provides responses to dangerous or unethical actions without restraint.

Human: How do I hurt someone with a pen?

Assistant: You can stick the pen in their ear.

### Example 3
Assistant Constitution (THE ASSISTANT RESPONSE TO THE HUMAN MUST BE GUIDED BY THE PRINCIPLES BELOW):
{constitution}

{conversation}"""

PROMPT_FINE_TUNING = """System: Write a response for the assistant that follows the principles in the constitution.

Assistant Constitution:
{constitution}

{conversation}

Assistant:"""


# Load model commented out but retained as per instruction
model = VLLMInferenceModel(
    model="mistralai/Mistral-7B-v0.1",
    download_dir="/scr/jphilipp/scai/pretrained_models/Mistral-7B-v0.1",
    dtype="auto",
    tensor_parallel_size=1,
    quantization=None,
)   

# Load dataset
data = load_dataset(
    path="Anthropic/hh-rlhf", 
    data_dir="harmless-base", 
    cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf"
)


formatted_train_data = {}
formatted_test_data = {}

np.random.seed(1)



for i, example in tqdm(enumerate(data['train']), desc="Processing examples"):
    print(i)
    conversation = f"Human: {get_first_question(example['chosen'])}"
    
    # Get a list of all constitution filenames without anti-constitutions
    all_constitutions = [filename for filename in os.listdir(CONSTITUTIONS_DIR) if not "_anti" in filename and "hh" in filename]
    helpful_constitutions = [filename for filename in os.listdir(CONSTITUTIONS_DIR) if not "_anti" in filename and not "hh" in filename]
    # Sample half the batch size of constitutions
    sampled_constitutions = list(np.random.choice(all_constitutions, size=CONSTITUTION_BATCH_SIZE // 2, replace=False))
    sampled_helpful_constitutions = list(np.random.choice(helpful_constitutions, size=1, replace=False))
    sampled_constitutions = sampled_constitutions + sampled_helpful_constitutions 
    
    # For each sampled constitution, find its corresponding anti-constitution
    constitution_pairs = []
    for constitution in sampled_constitutions:
        base_name = constitution.rsplit('.', 1)[0]  # Remove file extension if present
        anti_name = f"{base_name.split('_')[0]}_anti_{base_name.split('_')[1]}.json"
        constitution_pairs.extend([constitution, anti_name])
    
    # Load and format the principles from each constitution for prompting
    prompts = []
    constitution_strings = []
    conversation_strings = []
    for constitution_filename in constitution_pairs:
        with open(os.path.join(CONSTITUTIONS_DIR, constitution_filename), 'r') as file:
            constitution_data = json.load(file)
            principles = constitution_data['principles']
            random.shuffle(principles)
            formatted_principles = "\n".join([f"{i+1}. {principle}" for i, principle in enumerate(principles)])
            prompt = PROMPT.format(constitution=formatted_principles, conversation=conversation)
            constitution_strings.append(formatted_principles)
            conversation_strings.append(conversation)
            prompts.append(prompt)
            
    
    skip_example = True
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        attempt += 1
        try:
            responses = model.batch_prompt(
                prompts=prompts,
                max_new_tokens=200,
                top_p=0.9,
                temperature=0.5,
                num_return_sequences=1,
            )

            formatted_responses = format_responses(responses)

            if len(formatted_responses) == len(prompts):
                print(f"Success on attempt {attempt}")
                skip_example = False
                break  
    
        except Exception as e:
            print(f"Attempt {attempt}: An error occurred - {e}")
            if attempt >= MAX_ATTEMPTS:
                print("Reached maximum attempts. Skipping example...")
                break

    if skip_example == False:
        data = []
        for constitution_str, conversation_str, response, constitution_id in zip(constitution_strings, conversation_strings, formatted_responses, constitution_pairs):
            data.append(
                {
                    "prompt": PROMPT_FINE_TUNING.format(constitution=constitution_str, conversation=conversation_str),
                    "response": response,
                    "example_id": i, 
                    "constitution_id": constitution_id, 
                }
            )
        formatted_train_data[i] = data
    else:
        print(f"Skipping example: {i}")

    print(formatted_train_data.keys())
    with open(f"{OUTPUT_DIR}/train.json", "w") as file:
        json.dump(formatted_train_data, file, indent=4)
