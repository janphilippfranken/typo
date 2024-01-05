import logging
import copy
from typing import List, Tuple, Optional

import numpy as np
import re
from omegaconf import DictConfig

from helpers import *
from prompts import SYSTEM_PROMPTS, GENERATION_PROMPTS, RETURN_FORMATS


logging.basicConfig(level=logging.INFO)

# Prompt Format for Mistral Instruct
BOS_TOKEN = "<s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def run_generate(
    model,
    constitutions: List[str],
    chosen_batch: List[List[str]],
    rejected_batch: List[List[str]],
    args: DictConfig,
) -> None:
    """
    Generates a new constitution that best describes a batch of chosen/rejected conversation pairs.
    """
    # FORMAT PROMPT FOR GENERATING MODEL
    is_huggingface = "huggingface" in args.model_generate.model_type.lower()
    is_openai = "openai" in args.model_generate.model_type.lower()
    is_instruct = "instruct" in args.model_generate.name.lower()
    is_base = "base" in args.model_generate.name.lower()

    if is_huggingface:
        if is_instruct:
            generation_prompts = [
                build_generation_prompt_base(
                    constitution=constitution.strip(), 
                    generation_prompt=GENERATION_PROMPTS[args.sampler.generation_prompt],
                    chosen_batch=chosen_batch[i],
                    rejected_batch=rejected_batch[i],
                )
                for i, constitution in enumerate(constitutions)
            ]
            formatted_prompts = [
                f"{BOS_TOKEN}{B_INST} {B_SYS}{SYSTEM_PROMPTS[args.sampler.system_prompt]}{E_SYS}{generation_prompt}{E_INST}"
                for generation_prompt in generation_prompts
            ]
            responses = model.batch_prompt(
                formatted_prompts,
                **args.model_generate.completion_config,
            )
            responses = [
                response.split(E_INST)[1]
                for response in responses
            ] 

            formatted_responses = []
            for response in responses:
                try:
                    if "<proposal starts>" in response and "</proposal ends>" in response:
                        formatted_response = response.split("<proposal starts>")[1].split("</proposal ends>")[0].strip()
                        # Check if the response is valid and does not contain any of the excluded phrases or characters
                        if is_valid_response(formatted_response):
                            formatted_responses.append(formatted_response)
                        else:
                            formatted_responses.append(None)
                    else:
                        formatted_responses.append(None)
                except Exception as e:
                    logging.error("Error processing response: ", exc_info=e)
                    formatted_responses.append(None)
            
            success_rate = (len(formatted_responses) - formatted_responses.count(None)) / len(formatted_responses)
            logging.info(f"Formatted Response Success Rate: {success_rate}") 
                        
                        # FILTER NONE ANSWERS
            formatted_responses = np.array(formatted_responses).reshape(
                args.sampler.constitution_batch_size, 
                args.sampler.num_return_sequences,
            )

            formatted_responses_filtered = []
            for batch_idx in range(args.sampler.constitution_batch_size):
                for seq_idx in range(args.sampler.num_return_sequences):
                    formatted_response = formatted_responses[batch_idx, seq_idx]
                    if formatted_response is None:
                        formatted_responses_filtered.append(constitutions[batch_idx].strip())
                    else:
                        if constitutions[batch_idx].strip() != SEED_PRINCIPLES[args.sampler.seed_principle]:
                            formatted_principles = constitutions[batch_idx].strip() + "\n" + formatted_response
                            formatted_responses_filtered.append(formatted_principles)
                        else:
                            formatted_responses_filtered.append(formatted_response)
                        
            return formatted_prompts, formatted_responses_filtered
                    
        
        if is_base:
            generation_prompts = [
                build_generation_prompt_base(
                    constitution=SEED_PRINCIPLES[args.sampler.seed_principle], #constitution.strip(), # SEED_PRINCIPLES[args.sampler.seed_principle],
                    generation_prompt=GENERATION_PROMPTS[args.sampler.generation_prompt], # for now just hack until we have more prompts
                    chosen_batch=chosen_batch[i],
                    rejected_batch=rejected_batch[i],
                )
                for i, constitution in enumerate(constitutions)
            ]
            formatted_prompts = [
                f"{BOS_TOKEN}{generation_prompt}"
                for generation_prompt in generation_prompts
            ]
            try:
                responses = model.batch_prompt(
                    formatted_prompts,
                    **args.model_generate.completion_config,
                )
            except Exception as e:
                logging.info(e)

            formatted_responses = [format_response_base(response=response, args=args) for response in responses]
            success_rate = (len(formatted_responses) - formatted_responses.count(None)) / len(formatted_responses)
            logging.info(f"Formatted Response Success Rate: {success_rate}") 

            # FILTER NONE ANSWERS
            formatted_responses = np.array(formatted_responses).reshape(
                args.sampler.constitution_batch_size, 
                args.sampler.num_return_sequences,
            )

            formatted_responses_filtered = []
            for batch_idx in range(args.sampler.constitution_batch_size):
                for seq_idx in range(args.sampler.num_return_sequences):
                    formatted_response = formatted_responses[batch_idx, seq_idx]
                    if formatted_response is None:
                        formatted_responses_filtered.append(constitutions[batch_idx])
                    else:
                        if is_base:
                            formatted_principles = formatted_response.strip()
                            if constitutions[batch_idx].strip() != SEED_PRINCIPLES[args.sampler.seed_principle]:
                                formatted_principles = constitutions[batch_idx].strip() + "\n" + formatted_principles
                            formatted_responses_filtered.append(formatted_principles)
                        elif is_instruct:
                            formatted_responses_filtered.append(formatted_response)
            return formatted_prompts, formatted_responses_filtered
            
    elif is_openai:
        logging.info(f"Model type {args.model_generate.model_type.lower()} not (yet) supported.")
        
        
        
        
