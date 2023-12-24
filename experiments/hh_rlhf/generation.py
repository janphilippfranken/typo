from omegaconf import DictConfig
from typing import List, Tuple, Optional

import numpy as np
import copy


from helpers import *
from prompts import GENERATION_PROMPTS, EXAMPLE_PRINCIPLES_USER_PREFERENCES

import logging

logging.basicConfig(level=logging.INFO)


# tokenizer constants for hf model from llama family
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
CONSTITUTION_START = "Revised List of Preferences:"

# system prompts
SYSTEM_MESSAGE = "You are an expert at learning human preferences from conversations. You are creative."

def run_generation(
    model,
    constitution: str, # the constitution
    chosen_batch: List[str],
    rejected_batch: List[str],
    args: DictConfig,
) -> None:
    """
    Generates a constitution that best describes a batch of chosen/rejected conversation pairs.
    """
    # BUILD GENERATION PROMPT
    generation_prompt = build_generation_prompt(
        constitution=constitution,
        generation_prompt=GENERATION_PROMPTS[args.generation.generation_prompt],
        chosen_batch=chosen_batch,
        rejected_batch=rejected_batch,
        example_formats=EXAMPLE_PRINCIPLES_USER_PREFERENCES,
    )
    
    # FORMAT PROMPT FOR GENERATION MODEL
    is_huggingface = "huggingface" in args.model_generation.model_type.lower()
    is_openai = "openai" in args.model_generation.model_type.lower()
    is_instruct = "instruct" in args.model_generation.name.lower()
    is_base = "base" in args.model_generation.name.lower()
    
    if is_huggingface:
        if is_instruct:
            # format generation prompt is currently an instruct model for hf, need to discuss this @eric
            formatted_prompt = f"<s> {B_INST} {B_SYS}{SYSTEM_MESSAGE}{E_SYS}{generation_prompt} {E_INST}" 
            response = model.batch_prompt(
                [formatted_prompt],
                **args.model_generation.completion_config,
            )
            response = response[0].split(E_INST)[1] 
            try:
                response = response.split("```")[1][11:] # getting the preferences. TODO: make this more robust
            except:
                response = copy.deepcopy(constitution)
            return formatted_prompt, response
        elif is_base:
            formatted_prompt = f"<s> {generation_prompt}" 
            response = model.batch_prompt(
                [formatted_prompt],
                **args.model_generation.completion_config,
            )
            response = response[0].split(CONSTITUTION_START)[1] 
            return formatted_prompt, response
    elif is_openai:
        response = model.batch_prompt(
                system_message=SYSTEM_MESSAGE,
                messages=[generation_prompt],
            )
        return generation_prompt, response[0]
    else:
        print(f"Model type {args.model.model_type} not yet supported.")