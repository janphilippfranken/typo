import logging
from typing import List, Tuple, Optional

import numpy as np
from omegaconf import DictConfig

from helpers import *
from prompts import GENERATION_PROMPTS, EXAMPLE_PRINCIPLES_USER_PREFERENCES


logging.basicConfig(level=logging.INFO)

# Prompt Format for Mistral Instruct
BOS_TOKEN = "<s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def run_generation(
    model,
    constitutions: List[str],
    chosen_batch: List[str],
    rejected_batch: List[str],
    args: DictConfig,
) -> None:
    """
    Generates a new constitution that best describes a batch of chosen/rejected conversation pairs.
    """
    # BUILD GENERATION PROMPT
    generation_prompts = [
        build_generation_prompt(
            constitution=constitution,
            generation_prompt=GENERATION_PROMPTS[args.generation.generation_prompt],
            chosen_batch=chosen_batch,
            rejected_batch=rejected_batch,
            example_formats=EXAMPLE_PRINCIPLES_USER_PREFERENCES,
        )
        for constitution in constitutions
    ]

    # FORMAT PROMPT FOR GENERATING MODEL
    is_huggingface = "huggingface" in args.model_generation.model_type.lower()
    is_openai = "openai" in args.model_generation.model_type.lower()

    if is_huggingface:
        formatted_prompts = [
            f"{BOS_TOKEN}{B_INST} {B_SYS}{args.generation.system_message}{E_SYS}{generation_prompt}{E_INST}"
            for generation_prompt in generation_prompts
        ]
        responses = model.batch_prompt(
            formatted_prompts,
            **args.model_generation.completion_config,
        )
        responses = [
            response.split(E_INST)[1]
            for response in responses
        ]
        formatted_responses = []
        for response_idx, response in enumerate(responses):
            try:
                formatted_responses.append(
                    response.split("```")[1][len(args.generation.code_block_start):],
                )
            except:
                formatted_responses.append(
                    constitutions[response_idx], # deepcopy prob not necessary?
                )
        return formatted_prompts, formatted_responses
    elif is_openai:
        responses = model.batch_prompt(
                system_message=args.generation.system_message,
                messages=generation_prompts,
            )
        return generation_prompts, responses
    else:
        print(f"Model type {args.model.model_type} not (yet) supported.")