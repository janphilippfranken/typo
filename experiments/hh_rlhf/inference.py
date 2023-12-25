from typing import List, Tuple

from omegaconf import DictConfig
from datasets import Dataset

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
    Evaluates the log probs of answers in chosen_batch and rejected_batch given a constiution. 
    Currently batching here at level of eval examples, not constitutions as during generation. 
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
    system_message = system_message.replace(
        args.generation.constitution_start, 
        args.generation.constitution_instruction,
    )

    is_base = "base" in args.model_inference.name
        
    if is_base:
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
        print(f"Model type {args.model_inference.name} not (yet) supported.")
        
    # GET LOG PROBS
    chosen_prompts_no_final_answer = remove_final_answer(
        chosen_prompts, 
        final_chosen_answers,
    )
    rejected_prompts_no_final_answer = remove_final_answer(
        rejected_prompts, 
        final_rejected_answers,
    )
    log_probs_chosen = model.batch_log_probs(
        answers=chosen_prompts,
        prompts=chosen_prompts_no_final_answer,
    )
    log_probs_rejected = model.batch_log_probs(
        answers=rejected_prompts,
        prompts=rejected_prompts_no_final_answer,
    )
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
    log_probs_chosen, log_probs_rejected = [], []

    for constitution in constitutions:
        try:
            log_prob_chosen, log_prob_rejected = run_inference(
                model=model_inference,
                system_message=constitution,
                chosen_batch=chosen_batch,
                rejected_batch=rejected_batch,
                args=args,
            )
        except:
            log_prob_chosen, log_prob_rejected = torch.tensor(-torch.inf), torch.tensor([0])
    
        log_probs_chosen.append(log_prob_chosen)
        log_probs_rejected.append(log_prob_rejected)
    
    return log_probs_chosen, log_probs_rejected