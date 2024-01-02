from typing import List, Optional

import re


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
    prompt = prompt.rsplit(final_answer, 1)[0]
    return prompt, final_answer


def build_eval_prompt(
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