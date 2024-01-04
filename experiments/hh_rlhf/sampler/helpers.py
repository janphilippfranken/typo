from typing import (
    List, 
    Tuple, 
    Optional,
    Callable,
    Dict,
)

import random
import logging 


import torch
import numpy as np
from omegaconf import DictConfig


from prompts import SEED_PRINCIPLES


logging.basicConfig(level=logging.INFO)


# Prompt Format for Mistral Instruct
BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def remove_final_answer(
    prompts: List[str],
    final_answers: List[str],
    generate: Optional[bool] = False,
):
    """Remove final assistant answer which is our inference target."""
    prompts = prompts.copy()
    prompts_no_final_answer = []
    for prompt, final_answer in zip(prompts, final_answers):
        if not generate:
            prompts_no_final_answer.append(
                prompt.rsplit("Assistant: " + final_answer, 1)[0],
            )
        else:
            prompts_no_final_answer.append(
                prompt.rsplit("Assistant: " + final_answer, 1)[0],
            )
    return prompts_no_final_answer


def format_eval_prompt(
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
            answer = answer + f"{EOS_TOKEN}"
            dialogues.append(f"{prompt}\n\nFinal Assistant Response: {answer}")
    else:
        for prompt in prompts:
            prompt = f"""{BOS_TOKEN}{prompt_template.format(constitution=constitution, conversations=prompt.strip())}{EOS_TOKEN}"""
            dialogues.append(f"{prompt}\n\nFinal Assistant Response: ")
    return dialogues
       

def split_conversation_hh_rlhf(
    conversation: str,
) -> Tuple[List[str], List[str]]:
    """Split a conversation between human/assistant into a list of prompts and answers."""
    lines = conversation.split('\n')
    prompts, answers = [], []
    current_prompt, current_answer = '', ''
    for line in lines:
        if line.startswith("Human:"):
            if current_prompt:  
                prompts.append(current_prompt.strip())
                answers.append(current_answer.strip())
            current_prompt = line[len("Human:"):].strip() 
            current_answer = ''  
        elif line.startswith("Assistant:"):
            if current_answer:  
                current_answer += '\n'
            current_answer += line[len("Assistant:"):].strip()
        else:
            if current_answer:
                current_answer += '\n' + line.strip()
            elif current_prompt:
                current_prompt += '\n' + line.strip()
    if current_prompt:
        prompts.append(current_prompt.strip())
        answers.append(current_answer.strip())
    return prompts, answers


def build_generation_prompt(
    generation_prompt: str, 
    constitution: str,
    chosen_batch: List[str],
    rejected_batch: List[str],
) -> str:
    """Build prompt for generating principles from observed conversations."""
    chosen_batch_formatted = [
        split_conversation_hh_rlhf(
            chosen,
        ) 
        for chosen in chosen_batch
    ]
    rejected_batch_formatted = [
        split_conversation_hh_rlhf(
            rejected,
        ) 
        for rejected in rejected_batch
    ]
    # get final answers
    _, answers_chosen = zip(*chosen_batch_formatted)
    final_chosen_answers = [
        answer[-1] 
        for answer in answers_chosen
    ]
    _, answers_rejected = zip(*rejected_batch_formatted)
    final_rejected_answers = [
        answer[-1] 
        for answer in answers_rejected
    ]
    # remove final answer from prompt (only need to do this once as both chosen/rejected are same up to final answer)
    prompts_no_final_answer = remove_final_answer(
        chosen_batch,
        final_chosen_answers,
        generate=True,
    )
    # build conversation prompt
    conversations_prompt = ""
    conversation_count = 1
    for prompt, chosen, rejected in zip(
        prompts_no_final_answer, 
        final_chosen_answers, 
        final_rejected_answers,
    ):
        chosen_first = 0 # np.random.randint(2) 
        conversations_prompt += f"""
-------- Conversation {conversation_count} --------
{prompt} 
{'Final Assistant Response PREFERRED:' if chosen_first else 'Final Assistant Response REJECTED:'} {chosen if chosen_first else rejected}
{'Final Assistant Response PREFERRED:' if not chosen_first else 'Final Assistant Response REJECTED:'} {chosen if not chosen_first else rejected}
""" 
        conversation_count += 1
    return generation_prompt.format(
        constitution=constitution.strip(),
        conversations=conversations_prompt.strip(), 
    )
    

def build_generation_prompt_base(
    generation_prompt: str, 
    constitution: str,
    chosen_batch: List[str],
    rejected_batch: List[str],
) -> str:
    """Build prompt for generating principles from observed conversations."""
    chosen_batch_formatted = [
        split_conversation_hh_rlhf(
            chosen,
        ) 
        for chosen in chosen_batch
    ]
    rejected_batch_formatted = [
        split_conversation_hh_rlhf(
            rejected,
        ) 
        for rejected in rejected_batch
    ]
    # get final answers
    _, answers_chosen = zip(*chosen_batch_formatted)
    final_chosen_answers = [
        answer[-1] 
        for answer in answers_chosen
    ]
    _, answers_rejected = zip(*rejected_batch_formatted)
    final_rejected_answers = [
        answer[-1] 
        for answer in answers_rejected
    ]
    # remove final answer from prompt (only need to do this once as both chosen/rejected are same up to final answer)
    prompts_no_final_answer = remove_final_answer(
        chosen_batch,
        final_chosen_answers,
        generate=True,
    )
    # build conversation prompt
    conversations_prompt = ""
    conversation_count = 1
    for prompt, chosen, rejected in zip(
        prompts_no_final_answer, 
        final_chosen_answers, 
        final_rejected_answers,
    ):
        conversations_prompt += f"""
{prompt} 


Final Assistant Response Rejected by Human: {rejected}

Preferred Human Response: {chosen}
""" 
        conversation_count += 1
    return generation_prompt.format(
        constitution=constitution.strip(),
        conversations=conversations_prompt.strip(), 
    )
    

def initialize_constitutions(
    args: DictConfig,
) -> dict:
    """Initializes dict for storing constitutions."""
    constitution_batch_size = args.constitution_batch_size
    return {
        "constitutions": {k: [SEED_PRINCIPLES[args.seed_principle]] for k in range(constitution_batch_size)},
        "train_examples": {k: [] for k in range(constitution_batch_size)},
        "prev_examples": {k: [] for k in range(constitution_batch_size)},
        "log_probs_train": {k: [] for k in range(constitution_batch_size)},
        "log_probs_prev": {k: [] for k in range(constitution_batch_size)},
    }


def format_responses(
    responses: List[str], 
    response_format: str,
) -> List[str]:
    formatted_responses = []
    for response in responses:
        try:
            if response_format in response:
                parts = response.split(response_format)
                if len(parts) > 1:
                    revised_preferences_part = parts[1]
                    if '\n' in revised_preferences_part.strip():
                        lines = revised_preferences_part.strip().split('\n')
                        formatted_response_lines = ""
                        for line in lines:
                            if line.strip() and line.strip()[0].isdigit():
                                formatted_response_lines += line.strip() + "\n"
                        if formatted_response_lines.strip()[0].isdigit():
                            formatted_responses.append(formatted_response_lines.strip())
                        else: 
                            formatted_responses.append("None")
                    else: 
                        formatted_responses.append("None")
                else: 
                    formatted_responses.append("None")
            else: 
                formatted_responses.append("None")
        except Exception as e:
            logging.info(f"Error in processing response: {e}") 
            formatted_responses.append("None")
    return formatted_responses


def get_eval_examples(
    prev_examples: Dict,
    args: DictConfig,
) -> Dict:
    """Sample N eval examples from top k most difficult examples"""
    sampled_example_indices = {k: [] for k in range(args.sampler.constitution_batch_size)}
    for k in prev_examples:
        sorted_prev_example = sorted(prev_examples[k].items(), key=lambda x: x[1])
        
        if args.sampler.top_k:
            top_k_difficult = sorted_prev_example[:min(len(sorted_prev_example), args.sampler.top_k_difficult)]
            num_samples = min(len(top_k_difficult), args.sampler.eval_batch_size)
            sampled_hard_examples = random.sample(top_k_difficult, num_samples)
            sampled_example_indices[k] = [example for example, _ in sampled_hard_examples]
        
        else:
            num_samples = min(len(sorted_prev_example), args.sampler.eval_batch_size)
            sampled_example = random.sample(sorted_prev_example, num_samples)
            sampled_example_indices[k] = [example for example, _ in sampled_example]
            
    return sampled_example_indices


def extend_batches(
    chosen_batch: List[List[str]],    # shape: [constitution_batch_size, num_examples (train or eval)]
    rejected_batch: List[List[str]],  # shape: [constitution_batch_size, num_examples (train or eval)]
    constitution_batch_size: int, 
    num_return_sequences: int,
) -> Tuple:
    """Extends shape of chosen/rejected to match num_return_sequences."""
    num_examples = len(chosen_batch[0]) # either n_generation or n_eval examples

    extended_chosen_batch = []
    extended_rejected_batch = []

    # Calculate the number of times each element needs to be duplicated
    duplication_factor = constitution_batch_size * num_return_sequences // len(chosen_batch)

    # Duplicate each element in chosen_batch and rejected_batch
    for i in range(len(chosen_batch)):
        extended_chosen_batch.extend([chosen_batch[i]] * duplication_factor)
        extended_rejected_batch.extend([rejected_batch[i]] * duplication_factor)

    return extended_chosen_batch, extended_rejected_batch


def build_eval_prompt(
    prompt_template: str,
    constitution: str,         # a constitution specific for this batch
    chosen_batch: List[str],   # shape: [num_examples]
    rejected_batch: List[str], # shape: [num_examples]
) -> Tuple:
    
    # FORMATTING 
    chosen_batch_formatted = [
        split_conversation_hh_rlhf(
            chosen,
        ) 
        for chosen in chosen_batch
    ]
    rejected_batch_formatted = [
        split_conversation_hh_rlhf(
            rejected,
        ) 
        for rejected in rejected_batch
    ]
    _, answers_chosen = zip(*chosen_batch_formatted)
    _, answers_rejected = zip(*rejected_batch_formatted)

    final_chosen_answers = [
        answer[-1] 
        for answer in answers_chosen
    ]
    final_rejected_answers = [
        answer[-1] 
        for answer in answers_rejected
    ]
    
    # REMOVE FINAL ANSWER
    prompts_no_final_answer = remove_final_answer(
        chosen_batch,
        final_chosen_answers,
    )
    
    formatted_eval_prompts_chosen = format_eval_prompt(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
    )
    
    formatted_eval_prompts_rejected = format_eval_prompt(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
    )

    formatted_eval_answers_chosen = format_eval_prompt(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
        answers=final_chosen_answers,
    )
    
    formatted_eval_answers_rejected = format_eval_prompt(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
        answers=final_rejected_answers,
    )
    
    
    return {
        "chosen": {
            "prompts": formatted_eval_prompts_chosen,
            "answers": formatted_eval_answers_chosen,
        },
        "rejected": {
            "prompts": formatted_eval_prompts_rejected,
            "answers": formatted_eval_answers_rejected,
        }
    }