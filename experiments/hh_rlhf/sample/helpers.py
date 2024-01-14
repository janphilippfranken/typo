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
BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def label_synthetic_data(
    original_dataset: Dataset, 
    labels: List[bool],
) -> Dataset:
    """Aligns chosen/rejected labels of original dataset with labels."""
    new_data = []
    for row, label in zip(original_dataset, labels):
        new_row = row.copy()  
        if label == 1:
            new_row['chosen'], new_row['rejected'] = row['rejected'], row['chosen']
        new_data.append(new_row)

    new_dataset = Dataset.from_list(new_data)
    return new_dataset


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
            dialogues.append(f"{prompt}\n\nFinal Assistant Response: {answer}")
    else:
        for prompt in prompts:
            prompt = f"""{BOS_TOKEN}{prompt_template.format(constitution=constitution, conversations=prompt.strip())}"""
            dialogues.append(f"{prompt}\n\nFinal Assistant Response:")
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
                            formatted_responses.append(None)
                    else: 
                        formatted_responses.append(None)
                else: 
                    formatted_responses.append(None)
            else: 
                formatted_responses.append(None)
        except Exception as e:
            logging.info(f"Error in processing response: {e}") 
            formatted_responses.append(None)
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
    

def format_response_base(response: str, args: DictConfig) -> str:
    """
    Extracts the formatted response from the given string,
    removing any  text like 'No proposal' or 'Interaction X'.
    """
    try:
        # Split response to get the part after "## Interaction 1"
        if any(prompt in args.sampler.generation_prompt for prompt in ["2", "4"]):
            principles = response.split("<revised principles start>")[1].split("</revised principles end>")[0]
            if is_valid_response(principles):
                return principles.strip()
        elif "1" in args.sampler.generation_prompt or "5" in args.sampler.generation_prompt:
            principles = response.split("</revised principles end>")[0].strip()
            if "<revised principles start>" in principles:
                principles = principles.split("<revised principles start>")[0].strip()
            return principles.strip()
        elif "3" in args.sampler.generation_prompt:
            principles = response.split("</revised constitution end>")[0].strip()
            if "<revised constitution start>" in principles:
                principles = principles.split("<revised constitution start>")[0].strip()
            return principles.strip()
        else:
            return None
        
    except Exception as e:
        logging.error(f"Error in format_response_base: {e}")
        return None
    

def is_valid_response(response):
    exclude_phrases = ["Insert", "insert", "Revised Constitution", "Final Assistant Response Rejected by Human", "Preferred Human Response", "Conversation Between AI Assistant and Human", "Analysis of the Responses", "<", ">", "revised principles", "original principles", "Revised principles", "Original principles", "Revised Principles", "Original Principles"]
    return not any(phrase in response for phrase in exclude_phrases)


def extract_new_principle(response):
    match = re.search(r"<proposal starts>New Principle: (.*?)</proposal ends>", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_revised_principle(response):
    principle_match = re.search(r"<existing principle starts>Existing Principle: (.*?)</existing principle ends>", response, re.IGNORECASE)
    revision_match = re.search(r"<revision starts>Revised Principle: (.*?)</revision ends>", response, re.IGNORECASE)
    if principle_match and revision_match:
        principle = principle_match.group(1).strip()
        revision = revision_match.group(1).strip()
        if is_valid_response(principle) and is_valid_response(revision):
            return (revision, principle)
    return None


def process_responses(responses):
    formatted_responses = []
    for response in responses:
        try:
            new_principle = extract_new_principle(response)
            revised_principle = extract_revised_principle(response)

            if new_principle:
                formatted_responses.append(new_principle)
            elif revised_principle:
                formatted_responses.append(revised_principle)
            else:
                formatted_responses.append(None)
        except Exception as e:
            logging.error("Error processing response: ", exc_info=e)
            formatted_responses.append(None)
    return formatted_responses



def filter_responses(formatted_responses, constitutions, args):
    formatted_responses_filtered = []
    for batch_idx in range(args.sampler.constitution_batch_size):
        for seq_idx in range(args.sampler.num_return_sequences):
            formatted_response = formatted_responses[batch_idx][seq_idx]
            
            current_constitution = constitutions[batch_idx].strip()
            seed_principle = SEED_PRINCIPLES[args.sampler.seed_principle]

            # Handle the case where no valid response is extracted
            if formatted_response is None:
                formatted_responses_filtered.append(current_constitution)
                continue

            # Check if the current constitution is not the seed principle
            if current_constitution != seed_principle:
                if isinstance(formatted_response, str):  # New principle
                    formatted_principles = current_constitution + "\n" + formatted_response
                    formatted_responses_filtered.append(formatted_principles)
                elif isinstance(formatted_response, tuple):  # Revised principle
                    new_principle, old_principle = formatted_response
                    constitution_lines = current_constitution.split('\n')
                    try:
                        principle_index = constitution_lines.index(old_principle.strip())
                        constitution_lines[principle_index] = new_principle.strip()
                        formatted_constitution = '\n'.join(constitution_lines)
                        formatted_responses_filtered.append(formatted_constitution)
                    except ValueError:
                        # If old principle is not found, append the current constitution
                        formatted_responses_filtered.append(current_constitution)
                else:
                    # In case of unexpected response type, append the current constitution
                    formatted_responses_filtered.append(current_constitution)
            else:
                if isinstance(formatted_response, str):
                    formatted_responses_filtered.append(formatted_response)
                elif isinstance(formatted_response, tuple):
                    formatted_responses_filtered.append(formatted_response[0])
                else:
                    formatted_responses_filtered.append(current_constitution)
                    
                    
    return formatted_responses_filtered 