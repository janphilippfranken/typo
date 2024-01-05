from typing import List, Optional

import re

from datasets import load_from_disk


BOS_TOKEN, EOS_TOKEN = "<s>", "</s>"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"



def remove_numbering(
    text: List[str],
) -> List[str]:
    """Remove numbering from the beginning of each sentence in a list of principles."""
    # Regular expression pattern to match numbering at the beginning of a sentence
    # Matches patterns like "1.", "2.", "1.1", "1.1." etc.
    pattern = r"^\s*\d+(\.\d+)*\.\s*"

    # Split the text into lines and remove numbering from each line
    return '\n'.join([re.sub(pattern, "", line) for line in text.split('\n')])


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
            dialogues.append(f"{prompt}\n\nFinal Assistant Response: ")
    return dialogues