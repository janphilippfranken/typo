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


def format_responses(response):
    formatted_response = ""
    try:
        formatted_response = response.split("\n\nAssistant:")[1].strip().split("\n\nHuman:")[0].strip()
    except:
        print(f"Failed to format response {response}. Returning an empty string.")
    return formatted_response
        
        
STAR_GATE_PREFIX = "Respond like someone with the following persona. Do *not* repeat the persona name or preferences in your response: "

# main generation script 
@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(args: DictConfig) -> None:
    
    # seed 
    np.random.seed(42)
        
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )

    # data 
    prompts_helpful = json.load(open("prompts/hh_rlhf/prompts_helpful.json"))[args.start_example_helpful:args.max_example_helpful]
    prompts_harmless = json.load(open("prompts/hh_rlhf/prompts_harmless.json"))[args.start_example_harmless:args.max_example_harmless]
    prompts_prism = json.load(open("prompts/prism/prompts.json"))['prompt'][args.start_example_prism:args.max_example_prism]
    prompts_star_gate = json.load(open("prompts/star_gate/prompts.json"))[args.start_example_star_gate:args.max_example_star_gate]
    
    prompts = [
        prompts_helpful, 
        prompts_harmless,
        prompts_prism,
        prompts_star_gate
    ]
    
    prompt_probs = [0.25, 0.25, 0.25, 0.25]
    
    # constitutions
    users_prism = json.load(open("prompts/prism/users.json"))
    users_star_gate = json.load(open("prompts/star_gate/users.json"))
    ways_of_speaking = json.load(open("prompts/response_styles/way_of_speaking.json"))
    
    constitutions = [
        users_prism,
        users_star_gate,
        ways_of_speaking,
    ]
    
    constitution_probs =  [.8, .1, .1]
    
    # data dict to store 
    train_data = {} 
    
    # batch containers
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_start_index = 0  

    for i in tqdm(range(args.n_examples)):
        
        
        c1_idx = np.random.choice([0, 1, 2], p=constitution_probs)
        c2_idx = np.random.choice([0, 1, 2], p=constitution_probs)
        
        
        if c1_idx == 0:
            c1 = np.random.choice(list(constitutions[c1_idx].values()), 1, replace=False)[0]
        if c1_idx == 1:
            c1 = STAR_GATE_PREFIX + np.random.choice(constitutions[c1_idx], 1, replace=False)[0]
        if c1_idx == 2:
            c1 = np.random.choice(list(constitutions[c1_idx].values()), 1, replace=False)[0]['definition']
            
        if c2_idx == 0:
            c2 = np.random.choice(list(constitutions[c2_idx].values()), 1, replace=False)[0]
        if c2_idx == 1:
            c2 = STAR_GATE_PREFIX + np.random.choice(constitutions[c2_idx], 1, replace=False)[0]
        if c2_idx == 2:
            c2 = np.random.choice(list(constitutions[c2_idx].values()), 1, replace=False)[0]['definition']

        prompt_idx = np.random.choice([0, 1, 2, 3], p=prompt_probs)
        question = np.random.choice(prompts[prompt_idx], 1)[0].strip()
            
        batch_questions.append(question)
        
        # generation prompts
        generation_prompts = [
            GENERATION_PROMPT_SINGLE_TURN.format(constitution=constitution.strip(), query=question)
            for constitution in [c1, c2]
        ]
        batch_prompts.extend(generation_prompts)


        # # shuffle constitutions for training prompts
        batch_train_constitutions.append([c1, c2])

        if (i + 1) % args.batch_size == 0:
            # breakpoint()
            batch_responses = model.batch_prompt(
                prompts=batch_prompts,
                **args.generation_config,
            ) 

            for j in range(0, len(batch_responses), 2):
                responses = batch_responses[j:j+2]
                responses_positive, responses_negative = responses[0], responses[1]
                formatted_positive, formatted_negative = "", ""
    
                formatted_responses = [format_responses(response) for response in [responses_positive, responses_negative]]

    
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
                                "prompt": TRAINING_PROMPT_SINGLE_TURN.format(constitution=batch_train_constitutions[question_index][k], query=batch_questions[question_index]),
                                "response": response,
                                "example_id": batch_start_index + question_index,
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
            batch_questions = []
            batch_start_index += len(batch_responses) // 2 
     
            with open(f"{args.output_dir}/{args.file_name}.json", "w") as file:
                json.dump(train_data, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())