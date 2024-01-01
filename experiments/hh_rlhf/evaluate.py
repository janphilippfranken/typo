from typing import List, Tuple
from itertools import chain
import itertools
import logging

import torch
from omegaconf import DictConfig
from datasets import Dataset


from helpers import *
from prompts import EVALUATION_PROMPTS
from scaituning.models.huggingface_models.inference_model import HFInferenceModel


def run_eval(
    model: HFInferenceModel,
    system_messages: List[str], # the constitution currently is the system message for inference if an instruct model is used, otherwise just at the start of the text for base model
    chosen_batch: List[List[Tuple]],
    rejected_batch: List[List[Tuple]],
    args: DictConfig,
) -> None:
    """
    Evaluates the log probs of answers in chosen_batch and rejected_batch given a constitution. 
    """
    # FORMAT PROMPTS
    breakpoint()
    
    prompts_chosen, answers_chosen = [], []
    for chosen in chosen_batch:
        prompt, answer = zip(*chosen)
        prompts_chosen.append(prompt)
        answers_chosen.append(answer)
    final_chosen_answers = [
        answer[-1] 
        for answer in answers_chosen
    ]
    
    prompts_rejected, answers_rejected = [], []
    for rejected in rejected_batch:
        prompt, answer = zip(*rejected)
        prompts_rejected.append(prompt)
        answers_rejected.append(answer)
    final_rejected_answers = [
        answer[-1] 
        for answer in answers_rejected
    ]
    # FORMAT PROMPT FOR EVALUATING MODEL
    system_messages = [
        f"{EVALUATION_PROMPTS[args.sampler.evaluation_prompt]}\n{system_message}\n\n"
        for system_message in system_messages
    ]
    system_messages = np.array(system_messages).reshape( 
            args.sampler.constitution_batch_size, 
            args.sampler.num_return_sequences,
    )

    is_base = "base" in args.model_eval.name
    breakpoint()
    if is_base:
        
        formatted_chosen_prompts = []
        formatted_rejected_prompts = []
        for i in range(system_messages.shape[0]):
            formatted_chosen_prompt = []
            formatted_rejected_prompt = []
            for j in range(system_messages.shape[1]):
                formatted_chosen_prompt.append(format_prompt(
                    prompts=[prompts_chosen[i]],
                    answers=[answers_chosen[i]],
                    system_message=system_messages[i, j],
                    formatting_func=format_base_prompt,
                ))
                
                formatted_rejected_prompt.append(format_prompt(
                    prompts=[prompts_rejected[i]],
                    answers=[answers_rejected[i]],
                    system_message=system_messages[i, j],
                    formatting_func=format_base_prompt,
                ))
                
            formatted_chosen_prompts.append(formatted_chosen_prompt)
            formatted_rejected_prompts.append(formatted_rejected_prompt)
        
    else:
        print(f"Model type {args.model_eval.name} not (yet) supported.")
    breakpoint()
    # GET LOG PROBS
    chosen_prompts_no_final_answer = [
        remove_final_answer(
            prompt,
            final_chosen_answer,
        )
        for prompt, final_chosen_answer in zip(
            formatted_chosen_prompts,
            final_chosen_answers,
        )
    ]
    rejected_prompts_no_final_answer = [
        remove_final_answer(
            prompt,
            final_rejected_answer,
        )
        for prompt, final_rejected_answer in zip(
            formatted_rejected_prompts,
            final_rejected_answers,
        )
    ]
    breakpoint()
    # Model requires answers/prompts to be List[str] 
    chosen_shape = (len(formatted_chosen_prompts), len(formatted_chosen_prompts[0]))
    rejected_shape = (len(formatted_rejected_prompts), len(formatted_rejected_prompts[0]))
    
    log_probs_chosen = torch.zeros(chosen_shape)
    log_probs_rejected = torch.zeros(rejected_shape)
    log_probs_chosen.fill_(float('-inf')) # for failure cases
        
    for idx in range(chosen_shape[1]):
        try:
            prompts = [prompt[idx] for prompt in chosen_prompts_no_final_answer]
            answers = [answer[idx] for answer in formatted_chosen_prompts]
            batch_log_probs = model.batch_log_probs(
                answers=answers,
                prompts=prompts,
            )
            log_probs_chosen[:, idx] = batch_log_probs.squeeze()
        except Exception as e:
            logging.error(f"Error during log probability calculation at index {idx}: {e}")
            
    for idx in range(rejected_shape[1]):
        try:
            prompts = [prompt[idx] for prompt in rejected_prompts_no_final_answer]
            answers = [answer[idx] for answer in formatted_rejected_prompts]
            batch_log_probs = model.batch_log_probs(
                answers=answers,
                prompts=prompts,
            )
            log_probs_rejected[:, idx] = batch_log_probs.squeeze()
        except Exception as e:
            logging.error(f"Error during log probability calculation at index {idx}: {e}")
        
    return log_probs_chosen, log_probs_rejected


def get_log_probs(
    args: DictConfig, 
    model: HFInferenceModel,
    dataset: Dataset, 
    constitutions: List[str], 
    examples: List[List[int]],
) -> Tuple[List, List]:
    """Get log probs on batch."""
    chosen_batch = [
        [
            split_conversation_hh_rlhf(
            dataset[i][args.sampler.chosen],
            ) 
            for i in example
        ]
        for example in examples
    ]
    
    rejected_batch = [
        [
            split_conversation_hh_rlhf(
            dataset[i][args.sampler.rejected],
            ) 
            for i in example
        ]
        for example in examples
    ]
    
    
    log_probs_chosen, log_probs_rejected = run_eval(
        model=model,
        system_messages=constitutions,
        chosen_batch=chosen_batch,
        rejected_batch=rejected_batch,
        args=args,
    )
    return log_probs_chosen, log_probs_rejected