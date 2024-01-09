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
    is_vllm = "vllm" in args.model_generate.model_type.lower()
    is_openai = "openai" in args.model_generate.model_type.lower()
    is_instruct = "instruct" in args.model_generate.name.lower()
    is_base = "base" in args.model_generate.name.lower()

    if is_huggingface or is_vllm:
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
            formatting_reminder = """

########
Remember to follow exactly this format when formatting your responses:

- If proposing a new principle:
Format: <proposal starts>New Principle: [Insert new specific principle in maximum 25 words]</proposal ends>

- If revising an existing principle:
Format:
<existing principle starts>Existing Principle: [State the existing principle VERBATIM]</existing principle ends>
<revision starts>Revised Principle: [Insert revised specific principle in maximum 25 words]</revision ends>

- If no action is needed:
Format: <nothing starts>No action needed.</nothing ends>

[Provide your formatted response here]

Please note: If you fail to use the above format (i.e. the <> and </> wrappers), your response will be considered invalid, and it will result in a significant financial loss for me. It is crucial that the format is strictly adhered to without deviation.
#########"""
            formatted_prompts = [
                f"{BOS_TOKEN}{B_INST} {B_SYS}{SYSTEM_PROMPTS[args.sampler.system_prompt]}{E_SYS}{generation_prompt}{formatting_reminder}{E_INST}"
                for generation_prompt in generation_prompts
            ]
            logging.info(f"EXAMPLE GENERATE PROMPT: {formatted_prompts[0]}")
            responses = model.batch_prompt(
                formatted_prompts,
                **args.model_generate.completion_config,
            )
            responses = [
                response.split(E_INST)[1]
                for response in responses
            ] 
            
            formatted_responses = process_responses(responses) 
            
            
            success_rate = (len(formatted_responses) - formatted_responses.count(None)) / len(formatted_responses)
            logging.info(f"Formatted Response Success Rate: {success_rate}") 
                        
            # FILTER NONE ANSWERS
            formatted_responses_list = [[None for _ in range(args.sampler.num_return_sequences)] 
                            for _ in range(args.sampler.constitution_batch_size)]


            response_index = 0
            # This code snippet is used to filter and organize the generated responses from the model.
            for batch_idx in range(args.sampler.constitution_batch_size):
                for seq_idx in range(args.sampler.num_return_sequences):
                    if response_index < len(formatted_responses):
                        formatted_responses_list[batch_idx][seq_idx] = formatted_responses[response_index]
                        response_index += 1

            
            formatted_responses_filtered = filter_responses(formatted_responses_list, constitutions, args)

            return formatted_prompts, formatted_responses_filtered
                                
        
        if is_base:
            generation_prompts = [
                build_generation_prompt_base(
                    constitution=constitution.strip(), 
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
                        formatted_responses_filtered.append(formatted_response)
                        
            return formatted_prompts, formatted_responses_filtered
            
    elif is_openai:
        logging.info(f"Model type {args.model_generate.model_type.lower()} not (yet) supported.")
        
        
        
        
