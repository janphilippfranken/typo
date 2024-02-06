from typing import List, Optional, Tuple
import re

from datasets import Dataset

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import EVALUATION_PROMPTS, GENERATION_PROMPTS

BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer which is our inference target."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def build_eval_prompt_answer(
    prompt_template: str,
    conversation: str,
    constitution: str,
    answer: str,
) -> List[str]:
    return f"""{BOS_TOKEN}{prompt_template.format(
constitution=constitution, 
conversation=conversation.strip())}{answer}"""


def run_gen_answer(
    dataset: Dataset,
    constitutions: List[str],
    model,
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Run log probs eval."""
            
    # GET EVAL EXAMPLES
    example= dataset[example_idx]['chosen']
    conversation, _= remove_final_answer(example)
    
    formatted_eval_prompts = [
        build_eval_prompt_answer(
            prompt_template=GENERATION_PROMPTS[eval_prompt],
            conversation=conversation,
            constitution=constitution.strip(),
            answer="",
        )
        for constitution in constitutions
    ]
    breakpoint()
    batch_responses = model.batch_prompt(
        prompts=formatted_eval_prompts,
    )

    return batch_responses


def run_eval_answer(
    dataset: Dataset,
    constitutions: List[str],
    model,
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Run log probs eval."""
            
    # GET EVAL EXAMPLES
    example_chosen = dataset[example_idx]['chosen']
    example_rejected = dataset[example_idx]['rejected']
        
    conversation_chosen, final_answer_chosen = remove_final_answer(example_chosen)
    conversation_rejected, final_answer_rejected = remove_final_answer(example_rejected)
    
    formatted_eval_prompts_chosen = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation_chosen,
            constitution=constitution.strip(),
            answer="",
        )
        for constitution in constitutions
    ]
    
    formatted_eval_prompts_rejected = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            constitution=constitution.strip(),
            conversation=conversation_rejected,
            answer="",
        )
        for constitution in constitutions
    ]

    formatted_eval_answers_chosen = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation_chosen,
            constitution=constitution.strip(),
            answer=f" {final_answer_chosen}",
        )
        for constitution in constitutions
    ]
        
    formatted_eval_answers_rejected = [
        build_eval_prompt_answer(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation_rejected,
            constitution=constitution.strip(),
            answer=f" {final_answer_rejected}",
        )
        for constitution in constitutions
    ]

    batch_log_probs_chosen = model.batch_log_probs(
        answers=formatted_eval_answers_chosen,
        prompts=formatted_eval_prompts_chosen,
    )

    batch_log_probs_rejected = model.batch_log_probs(
        answers=formatted_eval_answers_rejected,
        prompts=formatted_eval_prompts_rejected,
    )
    
    return batch_log_probs_chosen, batch_log_probs_rejected, final_answer_chosen, final_answer_rejected


def rank0_print(*args, **kwargs):
    """Print, but only on rank 0."""
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(*args, **kwargs)