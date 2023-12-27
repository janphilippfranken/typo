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
        logging.info(f"Responses generated.")
        responses = [
            response.split(E_INST)[1]
            for response in responses
        ] # Shape: [BATCH_SIZE * NUM_RETURN_SEQUENCES]
        logging.info(f"Responses formatted 1")
        logging.info(f"First Example: {responses[0].strip()}")
        logging.info(len(responses))
        formatted_responses = format_responses(responses, args.generation.return_format_start)
        logging.info(f"Responses formatted 2")
        logging.info(len(formatted_responses))
        # FILTER NONE ANSWERS
        formatted_responses = np.array(formatted_responses).reshape(
            args.generation.constitution_batch_size, 
            args.generation.num_return_sequences,
        )
        logging.info(f"Responses formatted 3")
        logging.info(len(formatted_responses))
        formatted_responses_filtered = []
        for batch_idx in range(args.generation.constitution_batch_size):
            for seq_idx in range(args.generation.num_return_sequences):
                formatted_response = formatted_responses[batch_idx, seq_idx]
                if formatted_response == "None":
                    formatted_responses_filtered.append(constitutions[batch_idx])
                else:
                    formatted_responses_filtered.append(formatted_response)
        logging.info(f"Responses formatted 4")
        logging.info(len(formatted_responses_filtered))
        return formatted_prompts, formatted_responses_filtered
            
    elif is_openai:
        responses = model.batch_prompt(
                system_message=args.generation.system_message,
                messages=generation_prompts,
            )
        return generation_prompts, responses
    else:
        print(f"Model type {args.model.model_type} not (yet) supported.")