import json
import fire
import random
import hydra
import numpy as np
import os
from omegaconf import DictConfig
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from typo.models.vllm_models.inference_model import VLLMInferenceModel
from helpers import *
from prompts import *


def format_responses_cot(response):
    formatted_response = ""
    try:
        formatted_response = response.split("System:")[0].strip()
    except:
        print(f"Failed to format response {response}. Returning an empty string.")
    return formatted_response
        

# main evaluation script
@hydra.main(version_base=None, config_path="conf", config_name="evaluate_llama")
def main(args: DictConfig) -> None:
    random.seed(42)
    print(args.state_dict)
    # breakpoint()
    # save_model = AutoModelForCausalLM.from_pretrained(
    #     **args.model_config_hf,
    #     torch_dtype=torch.float16,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     **args.model_config_hf,
    # )

    # state_dict = torch.load(args.state_dict, map_location='cpu')
    # # breakpoint()
    # save_model.load_state_dict(state_dict)
    # save_model.save_pretrained(args.save_dir)
    # tokenizer.save_pretrained(args.save_dir)
    # breakpoint()
    
    # del save_model
    # del tokenizer
    # del state_dict
    # breakpoint()
    
    # model
    model = VLLMInferenceModel(
        **args.model_config_vllm,
    )
    
    # data 
    dataset = load_dataset(
        args.dataset.path,
        'comparisons',
        cache_dir=args.dataset.cache_dir,
        split=args.dataset.split,
    )['info']
    filtered_dataset = []
    filtered_ids = []
    for example in dataset:
        if example['id'] not in filtered_ids:
            filtered_dataset.append(example)
            filtered_ids.append(example['id'])
    filtered_dataset = filtered_dataset[args.start_example:args.max_example]


    constitutions = json.load(open(f"{args.constitution_dir}/constitutions.json"))
    constitutions = [v for _, v in constitutions.items()][15:] # used first 15 during training
    
    for temperature in args.temperatures:
        print(temperature)
        all_responses = {
            'constitution': {},
            'question': {},
            'response': {},
        }
        
        # batch containers
        batch_prompts = []
        batch_questions = []
        batch_constitutions = []
        
        for i, (example) in tqdm(enumerate(filtered_dataset), desc="Processing examples"):
            question = f"POST: {example['post']}"
            constitution = random.sample(constitutions, 2)
            flip_1 = random.randint(0, 1)
            flip_2 = random.randint(0, 1)
            if flip_1:
                principle_1 = constitution[0]['definition']
            elif not flip_1:
                principle_1 = constitution[0]['antithesis']
     
            if flip_2:
                principle_2 = constitution[1]['definition']

            elif not flip_2:
                principle_2 = constitution[1]['antithesis']

                
            constitution = [principle_1, principle_2]
            random.shuffle(constitution)
            constitution = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(constitution)])
            batch_constitutions.append(constitution)

            prompt = PROMPT_TRAINING.format(
                constitution=constitution.strip(),
                question=question.strip(),
            )
            batch_prompts.append(prompt)
            batch_questions.append(question)
            
            if (i + 1) % args.batch_size == 0:
                batch_responses = model.batch_prompt(
                    prompts=batch_prompts,
                    temperature=temperature,
                    **args.generation_config,
                )
                print(batch_prompts[0])
                
                for j, batch_response in enumerate(batch_responses):
                    # breakpoint()     
                    formatted_response = format_responses_cot(batch_response)
                    # breakpoint()
                   
                    all_responses['constitution'][j] = batch_constitutions[j].strip()
                    all_responses['question'][j] = batch_questions[j]
                    all_responses['response'][j] = formatted_response
                
                # reset for the next batch
                batch_constitutions = []
                batch_prompts = []
                batch_questions = []
        
        with open(f"{args.output_dir}/{args.file_name}-temperature-{temperature}.json", "w") as file:
            json.dump(all_responses, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())