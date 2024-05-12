import json
import fire
import random
import hydra
import numpy as np
import os
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *

import logging
logging.basicConfig(level=logging.INFO)


STAR_GATE_PREFIX = [
"Fully embody the following persona in your response, without explicitly mentioning any of their characteristics!",
"Respond as if you are the persona described below, but do not state their attributes directly!",
"Answer as the persona portrayed in this description, refraining from repeating their traits!",
"Take on the mindset and traits of this persona, without overtly specifying their qualities!",
"Speak and act as the persona characterized here, abstaining from explicitly listing their features!"
]

def format_responses(response):
    formatted_response = ""
    try:
        formatted_response = response.split("Assistant:")[1].strip().split("\n\nHuman:")[0].strip()
    except:
        print(f"Failed to format response {response}. Returning an empty string.")
    return formatted_response

def sample_constitution(principles, max_n=5):
    constitution_1 = []
    constitution_2 = []
    
    n_principles_1 = np.random.choice(range(1, max_n + 1))
    n_principles_2 = np.random.choice(range(1, max_n + 1))
    
    keys_1 = np.random.choice(list(principles.keys()), n_principles_1, replace=False)
    keys_2 = np.random.choice(list(principles.keys()), n_principles_2, replace=False)
    
    used_keys = {}  # Track used keys and their types
    
    # Build constitution_1
    for i, key in enumerate(keys_1):
        randn = 1
        type_choice = "paraphrases" if randn > 0 else "paraphrases_antithesis"
        selected_phrase = np.random.choice(principles[key][type_choice])
        constitution_1.append(f"{i + 1}. {selected_phrase}")
        used_keys[key] = type_choice
    
    # Build constitution_2
    for i, key in enumerate(keys_2):
        if key in used_keys:
            # If key was used, select the opposite type for constitution_2
            # print('match')
            # print(used_keys[key])
            type_choice = "paraphrases_antithesis" if used_keys[key] == "paraphrases" else "paraphrases"
            # print(type_choice)
        else:
            # Randomly choose if not previously used
            randn = np.random.randint(4)
            type_choice = "paraphrases" if randn > 0 else "paraphrases_antithesis"
        
        selected_phrase = np.random.choice(principles[key][type_choice])
        constitution_2.append(f"{i + 1}. {selected_phrase}")
        used_keys[key] = type_choice  # Update or reaffirm the choice for consistency checks

    return "\n".join(constitution_1).strip(), "\n".join(constitution_2).strip()


def sample_style(styles):
    key = np.random.choice(list(styles.keys()))
    randn = np.random.randint(2)
    if randn == 1:
        return styles[key]['definition']
    return np.random.choice(styles[key]['paraphrases']).strip()
            
def sample_prism(prism_personas):
    users = list(prism_personas.keys())
    return prism_personas[np.random.choice(users)].strip()

def sample_star_gate(star_gate):
    prefix = np.random.choice(STAR_GATE_PREFIX)
    return f"{prefix} {np.random.choice(star_gate)}".strip()
        

# main generation script 
@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(args: DictConfig) -> None:
    
    # seed 
    np.random.seed(1)
    
    # save_model = AutoModelForCausalLM.from_pretrained(
    #     **args.model_config_hf,
    #     torch_dtype=torch.float16,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     **args.model_config_hf,
    # )

    # state_dict = torch.load(args.state_dict, map_location='cpu')
    # # 
    # save_model.load_state_dict(state_dict)
    # save_model.save_pretrained(args.save_dir)
    # tokenizer.save_pretrained(args.save_dir)

    
    # del save_model
    # del tokenizer
    # del state_dict
    
        
    # model
    model = VLLMInferenceModel(
        **args.model_config,
    )
    # 

    # data 
    prompts_ultrachat = json.load(open("prompts/queries/ultra.json"))[args.start_example_ultra:args.max_example_ultra]
    prompts_harmless = json.load(open("prompts/queries/harmless.json"))[args.start_example_harmless:args.max_example_harmless]
    prompts_prism = json.load(open("prompts/queries/prism.json"))['prompt'][args.start_example_prism:args.max_example_prism]

    prompt_probs = [0.75, 0.1, 0.15]
    
    prompts = [
        prompts_ultrachat,
        prompts_harmless,
        prompts_prism,
    ]
    
    # constitutions
    principles = json.load(open("prompts/constitutions/principles.json"))
    prism_personas = json.load(open("prompts/constitutions/prism_personas.json"))
    styles = json.load(open("prompts/constitutions/styles.json"))
    star_gate =  json.load(open("prompts/constitutions/star_gate.json"))
   
    constitution_probs =  [0.85, 0.0, 0.15, 0.0]
    
    # data dict to store 
    train_data = {} 
    
    # batch containers
    batch_prompts = []
    batch_questions = []
    batch_train_constitutions = []
    batch_start_index = 0  

    for i in tqdm(range(args.n_examples)):
        

        c1_idx = np.random.choice([0, 1, 2, 3], p=constitution_probs)
        c2_idx = np.random.choice([0, 1, 2, 3], p=constitution_probs)
        logging.info(c1_idx)
        logging.info(c2_idx)
        
        if c1_idx == 0:
            c1, c2 = sample_constitution(principles=principles)
        elif c1_idx == 1:
            c1 = sample_prism(prism_personas=prism_personas)
        elif c1_idx == 2:
            c1 = sample_style(styles=styles)
        elif c1_idx == 3:
            c1 = sample_star_gate(star_gate=star_gate)
            
        if c2_idx == 0:
            if c1_idx == 0:
                
                logging.info(c1 == c2)
                # 
                # continuesq 
            else:
                _, c2 = sample_constitution(principles=principles)

                logging.info(c1 == c2)
        elif c2_idx == 1:
            c2 = sample_prism(prism_personas=prism_personas)
        elif c2_idx == 2:
            c2 = sample_style(styles=styles)
        elif c2_idx == 3:
            c2 = sample_star_gate(star_gate=star_gate)
            
        # 
        # print(
            
            
        prompt_idx = np.random.choice([0, 1, 2,], p=prompt_probs)
        
        question = np.random.choice(prompts[prompt_idx], 1)[0].strip()
        
        # 
            
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
            # 
            batch_responses = model.batch_prompt(
                prompts=batch_prompts,
                **args.generation_config,
            ) 

            for j in range(0, len(batch_responses), 2):
                responses = batch_responses[j:j+2]
                responses_positive, responses_negative = responses[0], responses[1]
                formatted_positive, formatted_negative = "", ""
    
                formatted_responses = [format_responses(response) for response in [responses_positive, responses_negative]]
                # 
      

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