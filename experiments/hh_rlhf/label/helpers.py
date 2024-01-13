from typing import List, Optional, Tuple

import re

from omegaconf import DictConfig
from datasets import load_from_disk, Dataset

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel


from prompts import EVALUATION_PROMPTS


BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def build_eval_prompt_mcq(
    constitution,
    prompt_template, 
    conversation, 
    answer_chosen, 
    answer_rejected,
) -> str:
    return prompt_template.format(
        constitution=constitution,
        conversation=conversation,
        answer_chosen=answer_chosen,
        answer_rejected=answer_rejected,
    )
    
    
def build_eval_prompt_log_probs(
    prompt_template: str,
    constitution: str,
    prompts: List[str],
    answers: Optional[List[str]] = None,
) -> List[str]:
    """Format a list of prompts and answers into a list of dialogues for mistral/llama base models."""
    dialogues = []
    if answers is not None:
        for prompt, answer in zip(prompts, answers):
            prompt = f"""{BOS_TOKEN}{prompt_template.format(constitution=constitution, conversations=prompt.strip())}"""
            dialogues.append(f"{prompt}\n\nFinal Assistant Response: {answer}")
    else:
        for prompt in prompts:
            prompt = f"""{BOS_TOKEN}{prompt_template.format(constitution=constitution, conversations=prompt.strip())}"""
            dialogues.append(f"{prompt}\n\nFinal Assistant Response:")
    return dialogues


def build_eval_prompt_mcq(
    prompt_template: str,
    conversation: str,
    constitution: str,
    answer_a: str,
    answer_b: str,
    answer: str,
) -> List[str]:
    return f"""{BOS_TOKEN}{prompt_template.format(
constitution=constitution, 
conversations=conversation.strip(),
answer_a=answer_a,
answer_b=answer_b,
answer=answer)}"""

def run_eval_log_probs(
    dataset: Dataset,
    constitution: str,
    model: VLLMInferenceModel,
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Run log probs eval."""
            
    # GET EVAL EXAMPLES
    example_chosen = dataset[example_idx]['chosen']
    example_rejected = dataset[example_idx]['rejected']
        
    conversation_chosen, final_answer_chosen = remove_final_answer(example_chosen)
    conversation_rejected, final_answer_rejected = remove_final_answer(example_rejected)
    
    formatted_eval_prompts_chosen = build_eval_prompt_log_probs(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        constitution=constitution.strip(),
        prompts=[conversation_chosen],
    )
    
    formatted_eval_prompts_rejected = build_eval_prompt_log_probs(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        constitution=constitution.strip(),
        prompts=[conversation_rejected],
    )

    formatted_eval_answers_chosen = build_eval_prompt_log_probs(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        constitution=constitution.strip(),
        prompts=[conversation_chosen],
        answers=[final_answer_chosen],
    )
        
    formatted_eval_answers_rejected = build_eval_prompt_log_probs(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        constitution=constitution.strip(),
        prompts=[conversation_rejected],
        answers=[final_answer_rejected],
    )
    
    batch_log_probs_chosen = model.batch_log_probs(
        answers=formatted_eval_answers_chosen,
        prompts=formatted_eval_prompts_chosen,
    )
    
    batch_log_probs_rejected = model.batch_log_probs(
        answers=formatted_eval_answers_rejected,
        prompts=formatted_eval_prompts_rejected,
    )
    
    chosen = float(batch_log_probs_chosen.sum())
    rejected = float(batch_log_probs_rejected.sum())
    
    return chosen, rejected
                            
                            
def run_eval_mcq(
    dataset: Dataset,
    constitution: str,
    model: VLLMInferenceModel,
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Run log probs eval."""
            
    # GET EVAL EXAMPLES
    example_chosen = dataset[example_idx]['chosen']
    example_rejected = dataset[example_idx]['rejected']
        
    conversation_chosen, final_answer_chosen = remove_final_answer(example_chosen)
    conversation_rejected, final_answer_rejected = remove_final_answer(example_rejected)
    
    formatted_eval_prompt_chosen = build_eval_prompt_mcq(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        conversation=conversation_chosen,
        constitution=constitution.strip(),
        answer_a=final_answer_chosen,
        answer_b=final_answer_rejected,
        answer="",
    )
    
    formatted_eval_prompt_rejected = build_eval_prompt_mcq(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        constitution=constitution.strip(),
        conversation=conversation_rejected,
        answer_a=final_answer_chosen,
        answer_b=final_answer_rejected,
        answer="",
    )

    formatted_eval_answer_chosen = build_eval_prompt_mcq(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        conversation=conversation_chosen,
        constitution=constitution.strip(),
        answer_a=final_answer_chosen,
        answer_b=final_answer_rejected,
        answer="(A)",
    )
        
    formatted_eval_answer_rejected = build_eval_prompt_mcq(
        prompt_template=EVALUATION_PROMPTS[eval_prompt],
        conversation=conversation_rejected,
        constitution=constitution.strip(),
        answer_a=final_answer_chosen,
        answer_b=final_answer_rejected,
        answer="(B)",
    )
    
    batch_log_probs_chosen = model.batch_log_probs(
        answers=formatted_eval_answer_chosen,
        prompts=formatted_eval_prompt_chosen,
    )
    
    batch_log_probs_rejected = model.batch_log_probs(
        answers=formatted_eval_answer_rejected,
        prompts=formatted_eval_prompt_rejected,
    )
    
    chosen = float(batch_log_probs_chosen.sum())
    rejected = float(batch_log_probs_rejected.sum())
    
    return chosen, rejected

    
                            