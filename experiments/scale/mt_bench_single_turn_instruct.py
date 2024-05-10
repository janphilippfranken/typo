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

from typing import Optional
import logging
logging.basicConfig(level=logging.INFO)

def load_questions(question_file: str):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions
    return questions

def format_responses(response):
    formatted_response = ""
    try:
        formatted_response = response.split("Assistant:")[1].strip().split("\n\nHuman:")[0].strip()
    except:
        print(f"Failed to format response {response}. Returning an empty string.")
    return formatted_response


# main generation script 
@hydra.main(version_base=None, config_path="conf", config_name="mt_bench_instruct")
def main(args: DictConfig) -> None:
    
    model = VLLMInferenceModel(
        **args.model_config_vllm,
    )

    
    # data 
    questions = load_questions('mt_bench/mt_bench.json')


    for temperature in args.temperatures:
        
        all_responses = {
            'question': {},
            'response': {},
        }
        
        # batch containers
        batch_prompts = []
        batch_questions = []
        
        for i, _ in tqdm(enumerate(questions), desc="Processing examples"):
            question = questions[i]['turns'][0]
         
            messages = [
                {"role": "user", "content": question},
            ]

            input_ids = model.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            )

            prompt = model.tokenizer.batch_decode(input_ids)[0]
       
            batch_prompts.append(prompt)
            batch_questions.append(question)

            # breakpoint()
            # if i >= 0:
            if (i + 1) == len(questions):
                batch_responses = model.batch_prompt(
                    prompts=batch_prompts,
                    temperature=temperature,
                    is_instruct=True,
                    **args.generation_config,
                )
                
                for j, batch_response in enumerate(batch_responses):
                    # breakpoint()
                    # formatted_response = format_responses(batch_response)
                   
                    all_responses['question'][j] = batch_questions[j]
                    all_responses['response'][j] = batch_responses[j]
                
                # reset for the next batch
                batch_prompts = []
                batch_questions = []
        
        with open(f"{args.output_dir}/{args.file_name}-temperature-{temperature}.json", "w") as file:
            json.dump(all_responses, file, indent=4)


if __name__ == "__main__":
    fire.Fire(main())