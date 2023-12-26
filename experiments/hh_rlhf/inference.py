from typing import List, Tuple
from itertools import chain
import itertools
import logging

import torch
from omegaconf import DictConfig
from datasets import Dataset


from helpers import *
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


def run_inference(
    model: HFInferenceModel,
    system_messages: List[str], # the constitution currently is the system message for inference if an instruct model is used, otherwise just at the start of the text for base model
    chosen_batch: List[Tuple],
    rejected_batch: List[Tuple],
    args: DictConfig,
) -> None:
    """
    Evaluates the log probs of answers in chosen_batch and rejected_batch given a constitution. 
    """
    # FORMAT PROMPTS
    prompts_chosen, answers_chosen = zip(*chosen_batch)
    final_chosen_answers = [
        answer[-1] 
        for answer in answers_chosen
    ]
    prompts_rejected, answers_rejected = zip(*rejected_batch)
    final_rejected_answers = [
        answer[-1] 
        for answer in answers_rejected
    ]
    # FORMAT PROMPT FOR EVALUATING MODEL
    system_messages = [
        system_message.replace(
            args.generation.constitution_start, 
            args.generation.constitution_instruction,
        ) 
        for system_message in system_messages
    ]

    is_base = "base" in args.model_inference.name
        
    if is_base:
        formatted_chosen_prompts = [
            format_prompt(
                prompts=prompts_chosen,
                answers=answers_chosen,
                system_message=system_message,
                formatting_func=format_base_prompt,
            )
            for system_message in system_messages
        ]
        formatted_rejected_prompts = [
            format_prompt(
                prompts=prompts_rejected,
                answers=answers_rejected,
                system_message=system_message,
                formatting_func=format_base_prompt,
            )  
            for system_message in system_messages
        ]   
    else:
        print(f"Model type {args.model_inference.name} not (yet) supported.")
        
    # GET LOG PROBS
    chosen_prompts_no_final_answer = [
        remove_final_answer(
            prompt,
            final_chosen_answers,
        )
        for prompt in formatted_chosen_prompts
    ]
    rejected_prompts_no_final_answer = [
        remove_final_answer(
            prompt,
            final_rejected_answers,
        )
        for prompt in formatted_rejected_prompts
    ]
    # Model requires answers/prompts to be List[str] 
    chosen_shape = (len(formatted_chosen_prompts), len(formatted_chosen_prompts[0]))
    rejected_shape = (len(formatted_rejected_prompts), len(formatted_rejected_prompts[0]))
    
    log_probs_chosen = torch.zeros(chosen_shape)
    log_probs_rejected = torch.zeros(rejected_shape)
    log_probs_chosen.fill_(float('-inf')) # for failure cases

    try:
        log_probs_chosen = model.batch_log_probs(
            answers=list(itertools.chain(*formatted_chosen_prompts)),
            prompts=list(itertools.chain(*chosen_prompts_no_final_answer)),
        ).view(chosen_shape)
    except Exception as e:
        logging.error(f"Error during log probability calculation: {e}")
                
    try:
        log_probs_rejected = model.batch_log_probs(
            answers=list(itertools.chain(*formatted_rejected_prompts)),
            prompts=list(itertools.chain(*rejected_prompts_no_final_answer)),
        ).view(rejected_shape)
    except Exception as e:
        logging.error(f"Error during log probability calculation: {e}")
        
    return log_probs_chosen, log_probs_rejected


def get_log_probs(
    args: DictConfig, 
    model: HFInferenceModel,
    dataset: Dataset, 
    constitutions: List[str], 
    examples: List[int],
) -> Tuple[List, List]:
    """Get log probs on batch."""
    chosen_batch = [
        split_conversation_hh_rlhf(
            dataset[i][args.generation.chosen],
        ) 
        for i in examples
    ]
    rejected_batch = [
        split_conversation_hh_rlhf(
            dataset[i][args.generation.rejected],
        ) 
        for i in examples
    ]
    log_probs_chosen, log_probs_rejected = run_inference(
        model=model,
        system_messages=constitutions,
        chosen_batch=chosen_batch,
        rejected_batch=rejected_batch,
        args=args,
    )
    return log_probs_chosen, log_probs_rejected