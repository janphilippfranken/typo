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
    constitutions: List[str],         # shape: [constitution_batch_size * num_return_sequences]  or [constitution_batch_size]
    chosen_batch: List[List[str]],    # shape: [constitution_batch_size, num_examples]
    rejected_batch: List[List[str]],  # shape: [constitution_batch_size, num_examples]
    args: DictConfig,
) -> None:
    """
    Evaluates the log probs of answers in chosen_batch and rejected_batch given a constitution. 
    """   
    # FORMAT PROMPTS TO MATCH NUM_RETURN_SEQUENCES
    extended_chosen_batch, extended_rejected_batch = extend_batches(
        chosen_batch=chosen_batch, 
        rejected_batch=rejected_batch, 
        constitution_batch_size=args.sampler.constitution_batch_size,
        num_return_sequences=args.sampler.num_return_sequences,
    )
    
    evaluation_prompts = [ 
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[args.sampler.evaluation_prompt],
            constitution=constitution.strip(),
            chosen_batch=extended_chosen_batch[i] if len(constitutions) > len(chosen_batch) else chosen_batch[i],
            rejected_batch=extended_rejected_batch[i] if len(constitutions) > len(rejected_batch) else rejected_batch[i],
        )
        for i, constitution in enumerate(constitutions)
    ]
    # logging.info(f"EXAMPLE EVAL PROMPT: {evaluation_prompts[0]}")

    evaluation_prompts_chosen = [evaluation_prompt["chosen"] for evaluation_prompt in evaluation_prompts] 
    evaluation_prompts_rejected = [evaluation_prompt["rejected"] for evaluation_prompt in evaluation_prompts]
    
    eval_shape = (len(constitutions), len(chosen_batch[0]))
    
    log_probs_chosen = torch.zeros(eval_shape)
    log_probs_rejected = torch.zeros(eval_shape)
    log_probs_chosen.fill_(float('-inf'))  # for failure cases
        
        
    # GET LOG PROBS
    for idx in range(eval_shape[1]):
        try:
            prompts = [prompt['prompts'][idx] for prompt in evaluation_prompts_chosen]
            answers = [prompt['answers'][idx] for prompt in evaluation_prompts_chosen]
            batch_log_probs = model.batch_log_probs(
                answers=answers,
                prompts=prompts,
            )
            log_probs_chosen[:, idx] = batch_log_probs.squeeze()
        except Exception as e:
            logging.error(f"Error during log probability calculation at index {idx}: {e}")
            
    for idx in range(eval_shape[1]):
        try:
            prompts = [prompt['prompts'][idx] for prompt in evaluation_prompts_rejected]
            answers = [prompt['answers'][idx] for prompt in evaluation_prompts_rejected]
            batch_log_probs = model.batch_log_probs(
                answers=answers,
                prompts=prompts,
            )
            log_probs_rejected[:, idx] = batch_log_probs.squeeze()
        except Exception as e:
            logging.error(f"Error during log probability calculation at index {idx}: {e}")
        
    return log_probs_chosen, log_probs_rejected