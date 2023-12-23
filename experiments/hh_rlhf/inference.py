from omegaconf import DictConfig
from typing import List, Tuple

from helpers import *

from scaituning.models.huggingface_models.inference_model import HFInferenceModel


def run_inference(
    model: HFInferenceModel,
    system_message: str, # the constitution becomes the system message for inference if an instruct model is used, otherwise just in text for base model
    chosen_batch: List[Tuple],
    rejected_batch: List[Tuple],
    args: DictConfig,
) -> None:
    """
    Evaluates the log probs of answers in chosen_batch and 
    rejected_batch given a constiution.
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
    # check base or instruct model
    is_base = "base" in args.model_inference.name
    if is_base:
        # format chosen prompts
        chosen_prompts = format_prompt(
            prompts=prompts_chosen,
            answers=answers_chosen,
            system_message=system_message,
            formatting_func=format_base_prompt,
        )
        rejected_prompts = format_prompt(
            prompts=prompts_rejected,
            answers=answers_rejected,
            system_message=system_message,
            formatting_func=format_base_prompt,
        )
    else:
        # would use format_instruct_prompt as func here if implemented
        raise NotImplementedError(f"{args.model_inference.name} not implemented yet")
        
    # format prompts to not include final answer
    chosen_prompts_no_final_answer = remove_final_answer(
        chosen_prompts, 
        final_chosen_answers,
    )
    rejected_prompts_no_final_answer = remove_final_answer(
        rejected_prompts, 
        final_rejected_answers,
    )
    
    # GET LOG_PROBS
    log_probs_chosen = model.batch_log_probs(
        answers=chosen_prompts,
        prompts=chosen_prompts_no_final_answer,
    )
    log_probs_rejected = model.batch_log_probs(
        answers=rejected_prompts,
        prompts=rejected_prompts_no_final_answer,
    )
    
    return log_probs_chosen, log_probs_rejected