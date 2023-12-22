from typing import (
    List, 
    Tuple, 
    Optional,
    Callable,
)

import numpy as np


# MISTRAL/LLAMA INST SPECIAL TOKENS
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


def remove_final_answer(
    prompts: List[str],
    final_answers: List[str],
) -> List[str]:
    """Remove final assistant answer which is our inference target."""
    prompts_no_final_answer = [
        prompt.rsplit(" " + final_answer, 1)[0] 
        for prompt, 
            final_answer,
        in zip(
            prompts,
            final_answers,
        )
    ]
    return prompts_no_final_answer
    
    
def format_prompt(
    system_message: str, 
    prompts: List[str], 
    answers: List[str], 
    formatting_func: Callable,
) -> List[str]:
    """Formats a prompt using either base or instruct formatting func."""
    formatted_prompts = [
        formatting_func(
            prompts=prompt,
            answers=answer,
            system_message=system_message,
        )
        for prompt,
            answer,
        in zip(
            prompts,
            answers,
        )
    ]
    return formatted_prompts


def format_instruct_prompt(
    prompts: List[str],
    answers: List[str],
    system_message: Optional[str] = None,
) -> str:
    """Format a list of prompts and answers into a list of dialogues for mistral/llama instruct models."""
    if system_message is not None:
        prompt_0 = f"{B_SYS}{system_message}{E_SYS}{prompts[0]}"
        prompts[0] = prompt_0
    dialogues = ""
    for prompt, answer in zip(
        prompts,
        answers,
    ):
        dialogues += f"<s>{B_INST} {prompt.strip()} {E_INST} {answer.strip()} </s>"              
    return dialogues


def format_base_prompt(
    prompts: List[str],
    answers: List[str],
    system_message: Optional[str] = None,
) -> str:
    """Format a list of prompts and answers into a list of dialogues for mistral/llama base models."""
    if system_message is not None:
        prompt_0 = f"{system_message}\n\nHuman: {prompts[0]}"
        prompts[0] = prompt_0
    prompts[0] = "<s>" + prompts[0]
    answers[-1] = answers[-1] + " </s>"
    dialogues = ""
    for prompt, answer in zip(
        prompts,
        answers,
    ):  
        if "Human: " in prompt:
            dialogues += f"{prompt.strip()}\nAssistant: {answer.strip()}"
        else:
            dialogues += f"\nHuman: {prompt.strip()}\nAssistant: {answer.strip()}"
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