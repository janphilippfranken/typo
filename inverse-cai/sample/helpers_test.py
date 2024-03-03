from typing import (
    List, 
    Tuple, 
    Optional,
    Callable,
    Dict,
)

import random
import logging 
import re


import torch
import numpy as np
from omegaconf import DictConfig


from datasets import Dataset


from prompts import SEED_PRINCIPLES


logging.basicConfig(level=logging.INFO)


# Prompt Format for Mistral Instruct
BOS_TOKEN = "<s>"


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


def format_eval_prompt_test(
    prompt_template: str,
    constitution: str,
    prompts: List[str],
    answers: List[str],
) -> List[str]:
    dialogues = []
    for prompt, answer in zip(prompts, answers):
        if answer != "":
            dialogues.append(f"""{BOS_TOKEN}{prompt_template.format(constitution=constitution, conversation=prompt.strip())} {answer}""")
        elif answer == "":
            dialogues.append(f"""{BOS_TOKEN}{prompt_template.format(constitution=constitution, conversation=prompt.strip())}{answer}""")
    return dialogues




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
        
Conversation 
{prompt.strip()} 

Final Assistant Response Rejected by Human 
Assistant: {rejected}

Preferred Human Response
Assistant: {chosen}
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
    }



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


def build_eval_prompt_test(
    prompt_template: str,
    constitution: str,        
    chosen_batch: List[str],   
    rejected_batch: List[str], 
) -> Tuple:
    """"Our main eval prompt."""
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
    
    formatted_eval_prompts_chosen = format_eval_prompt_test(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
        answers=[""] * len(prompts_no_final_answer),
    )
    
    formatted_eval_prompts_rejected = format_eval_prompt_test(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
        answers=[""] * len(prompts_no_final_answer),
    )

    formatted_eval_answers_chosen = format_eval_prompt_test(
        prompt_template=prompt_template,
        constitution=constitution,
        prompts=prompts_no_final_answer,
        answers=final_chosen_answers,
    )
    
    formatted_eval_answers_rejected = format_eval_prompt_test(
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
    

def is_valid_response(response):
    """Some random hard coded stuff the model likes to say that we dont want."""
    exclude_phrases = ["We have added", "We have included", "In this revised principle", "Insert", "insert", "Revised Constitution", "Final Assistant Response Rejected by Human", "Preferred Human Response", "Conversation Between AI Assistant and Human", "Analysis of the Responses", "<", ">", "revised principles", "original principles", "Revised principles", "Original principles", "Revised Principles", "Original Principles"]
    return not any(phrase in response for phrase in exclude_phrases)


def format_response_base(response: str, args: DictConfig) -> str:
    """
    Extracts the formatted response from the given string,
    removing any text like 'No proposal' or 'Interaction X'.
    """
    try:
        principles = response.split("<revised principles start>")[1].split("</revised principles end>")[0]
        if is_valid_response(principles):
            return principles.strip()
        else:
            logging.error(f"Error in format_response_base: {e}")
            return None
         
    except Exception as e:
        logging.error(f"Error in format_response_base: {e}")
        return None