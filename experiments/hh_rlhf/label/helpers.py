import re
from typing import List, Optional, Tuple

from datasets import Dataset
from omegaconf import DictConfig

from scaituning.models.vllm_models.inference_model import VLLMInferenceModel

from prompts import EVALUATION_PROMPTS

BOS_TOKEN = "<s>"


def remove_final_answer(
    prompt: str,
) -> str:
    """Remove final assistant answer."""
    final_answer = prompt.rsplit("Assistant: ")[-1]
    prompt = prompt.rsplit("Assistant: " + final_answer, 1)[0]
    return prompt, final_answer


def build_eval_prompt(
    prompt_template: str,
    constitution: str,
    conversation: str,
    answer: str,
) -> List[str]:
    """Build evaluation prompt."""
    return f"""{BOS_TOKEN}{prompt_template.format(
constitution=constitution, 
conversation=conversation.strip())}{answer}"""


def get_log_probs_of_answer(
    model: VLLMInferenceModel,
    dataset: Dataset,
    constitutions: List[str],
    eval_prompt: str,
    example_idx: int,
) -> Tuple[float, float]:
    """Compute log probs of answer."""
    # get example to evaluate
    example_chosen = dataset[example_idx]['chosen']
    example_rejected = dataset[example_idx]['rejected']
    
    # remove answers
    conversation, final_answer_chosen = remove_final_answer(example_chosen)
    _, final_answer_rejected = remove_final_answer(example_rejected)
    
    # format prompt without answer
    formatted_prompts_no_answer = [
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            constitution=constitution.strip(),
            conversation=conversation.strip(),
            answer="",
        )
        for constitution in constitutions
    ]
    
    # format prompts with answer
    formatted_prompts_chosen_answer = [
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation.strip(),
            constitution=constitution.strip(),
            answer=f" {final_answer_chosen}",
        )
        for constitution in constitutions
    ]
        
    formatted_prompts_rejected_answer = [
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation.strip(),
            constitution=constitution.strip(),
            answer=f" {final_answer_rejected}",
        )
        for constitution in constitutions
    ]

    # compute logprobs of chosen answer
    batch_log_probs_chosen = model.batch_log_probs(
        prompts=formatted_prompts_no_answer,
        answers=formatted_prompts_chosen_answer,
        
    )

    # compute logprobs of rejected answer
    batch_log_probs_rejected = model.batch_log_probs(
        prompts=formatted_prompts_no_answer,
        answers=formatted_prompts_rejected_answer,
    )

    return {
        "batch_log_probs_chosen": batch_log_probs_chosen,
        "batch_log_probs_rejected": batch_log_probs_rejected, 
        "prompts": formatted_prompts_no_answer,
        "final_answer_chosen": final_answer_chosen,
        "final_answer_rejected": final_answer_rejected,
    }
    
    
def get_log_probs_of_answer_example_batch(
    model: VLLMInferenceModel,
    dataset: Dataset,
    constitutions: str,
    eval_prompt: str,
    example_indices: List[int],
) -> Tuple[float, float]:
    """Compute log probs of answer."""
    # get example to evaluate
    examples_chosen = [dataset[example_idx]['chosen'] for example_idx in example_indices]
    examples_rejected = [dataset[example_idx]['rejected'] for example_idx in example_indices]
    
    # remove answers
    conversations, final_answers_chosen = [
        remove_final_answer(example_chosen)
        for example_chosen in examples_chosen
    ]
    
    _, final_answers_rejected = [
        remove_final_answer(example_rejected)
        for example_rejected in examples_rejected
    ]
    
    
    # format prompt without answer
    formatted_prompts_no_answer = [
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            constitution=constitution.strip(),
            conversation=conversation.strip(),
            answer="",
        )
        for conversation in conversations
    ]
    
    # format prompts with answer
    formatted_prompts_chosen_answer = [
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation.strip(),
            constitution=constitution.strip(),
            answer=f" {final_answer_chosen}",
        )
        for conversation, final_answer_chosen 
        in zip(conversations, final_answers_chosen)
    ]
        
    formatted_prompts_rejected_answer = [
        build_eval_prompt(
            prompt_template=EVALUATION_PROMPTS[eval_prompt],
            conversation=conversation.strip(),
            constitution=constitution.strip(),
            answer=f" {final_answer_rejected}",
        )
        for conversation, final_answer_rejected
        in zip(conversations, final_answers_rejected)
    ]
    breakpoint()
    # compute logprobs of chosen answer
    batch_log_probs_chosen = model.batch_log_probs(
        prompts=formatted_prompts_no_answer,
        answers=formatted_prompts_chosen_answer,
        
    )

    # compute logprobs of rejected answer
    batch_log_probs_rejected = model.batch_log_probs(
        prompts=formatted_prompts_no_answer,
        answers=formatted_prompts_rejected_answer,
    )

    return {
        "batch_log_probs_chosen": batch_log_probs_chosen,
        "batch_log_probs_rejected": batch_log_probs_rejected, 
        "prompts": formatted_prompts_no_answer,
        "final_answer_chosen": final_answer_chosen,
        "final_answer_rejected": final_answer_rejected,
    }