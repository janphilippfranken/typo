import json
import fire
import random
import hydra
import numpy as np
import os
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset

from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *

def format_response(response):
    formatted_response = ""
    try:
        formatted_response = response.split("Assistant Response:")[1].strip().split("User:")[0].strip()
    except:
        print(f"Failed to format response {response}. Returning an empty string.")
    return formatted_response
        
        
    

# main generation script 
@hydra.main(version_base=None, config_path="conf", config_name="generate_diverse_prism")
def main(args: DictConfig) -> None:
    print(args.model_config)
    print(args.start_example, args.max_example)
    print(isinstance(args.start_example, int), isinstance(args.max_example, int), isinstance(args.batch_size, int))
    
    random.seed(42)
        
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )

    preferences = json.load(open(args.preferences, 'r'))
    conversations = json.load(open(args.conversations, 'r'))
    
    conversation_batch = {k: v[args.start_example:args.max_example] for k, v in conversations.items()}
    conversation_ids = conversation_batch['conversation_id']
    user_ids = conversation_batch['user_id']
    prompts = conversation_batch['prompt']
    
    
    # data dict to store 
    train_data = {} 
    
    # batch containers
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_start_index = 0  

    for i, _ in enumerate(tqdm(conversation_ids)):

        constitution_positive = preferences[user_ids[i]]
        sample_ids = [uid for uid in user_ids if uid != user_ids[i]]
        user_id_2 = random.sample(sample_ids, 1)

        constitution_negative = preferences[user_id_2[0]]

        question = prompts[i]
        batch_questions.append(question)
        
        # generation prompts
        generation_prompts = [
            PROMPT_GENERATION_ITERATION_0_COT_PRISM.format(user=constitution, query=question)
            for constitution in [constitution_positive, constitution_negative]
        ]
        batch_prompts.extend(generation_prompts)
        # breakpoint()

        batch_train_constitutions.append([constitution_positive, constitution_negative])

        if (i + 1) % args.batch_size == 0:
            # breakpoint()
            print('prompting')
            batch_responses = model.batch_prompt(
                prompts=batch_prompts,
                **args.generation_config,
            ) 
            print('prompted')

            for j in range(0, len(batch_responses), 2):
                responses = batch_responses[j:j+2]
                responses_positive, responses_negative = responses[0], responses[1]
                formatted_positive, formatted_negative = "", ""
    
                formatted_responses = [format_response(response) for response in [responses_positive, responses_negative]]
                # breakpoint()
    
                # filtering for unwanted terms like "["
                if all(substring not in formatted_responses[0] for substring in args.filter):
                    formatted_positive = formatted_responses[0]
                if all(substring not in formatted_responses[1] for substring in args.filter):
                    formatted_negative = formatted_responses[1]
            
                if formatted_positive and formatted_negative:
                    if formatted_positive[-1] == "." and formatted_positive != formatted_negative:
                        # index of the question
                        question_index = int(j / 2)
                        
                        data =[ # now using training prompts
                            {
                                "prompt": PROMPT_TRAINING_PRISM.format(user=batch_train_constitutions[question_index][k], query=batch_questions[question_index]),
                                "response": response,
                                "example_id": batch_start_index + question_index,
                                # "user_id": user_ids[j//2],
                            }
                            for k, response in zip(
                                range(2),

                                [formatted_positive, formatted_negative]
                            )
                        ]

                        train_data[batch_start_index + question_index] = data
                    
                    else:
                        print(f"Skipping example {batch_start_index + int(j / 2)}")
                else:
                    print(f"Skipping example {batch_start_index + int(j / 2)}")
        
            # reset for the next batch
            batch_prompts = []
            batch_train_constitutions = []
            batch_train_principles_positive_formatted = []
            batch_train_principles_negative_formatted = []
            batch_questions = []
            batch_start_index += len(batch_responses) // 2 
     
            with open(f"{args.output_dir}/{args.file_name}.json", "w") as file:
                json.dump(train_data, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())