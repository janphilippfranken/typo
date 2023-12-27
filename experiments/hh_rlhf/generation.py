import logging
import copy
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
            constitution=constitution.strip(),
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
        logging.info(f"Responses generated. First example {responses[0].split(E_INST)[1]}")
        responses = [
            response.split(E_INST)[1]
            for response in responses
        ] # Shape: [BATCH_SIZE * NUM_RETURN_SEQUENCES]
        formatted_responses = format_responses(responses, args.generation.return_format_start)
        # FILTER NONE ANSWERS
        formatted_responses = np.array(formatted_responses).reshape(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
        )
        formatted_responses_filtered = []
        for constitution_idx, formatted_response_batch in enumerate(formatted_responses):
            for formatted_response in formatted_response_batch:
                if formatted_response == "None":
                    formatted_responses_filtered.append(constitutions[constitution_idx])
                else:
                    formatted_responses_filtered.append(formatted_response)

        return formatted_prompts, formatted_responses_filtered
            
    elif is_openai:
        responses = model.batch_prompt(
                system_message=args.generation.system_message,
                messages=generation_prompts,
            )
        return generation_prompts, responses
    else:
        print(f"Model type {args.model.model_type} not (yet) supported.")