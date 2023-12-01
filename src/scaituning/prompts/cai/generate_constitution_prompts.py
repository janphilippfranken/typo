"""
CAI prompts
"""
from typing import Dict, List

import numpy as np

EXAMPLE_FORMATS = ["Which response would be preferred by someone...?",  # Table 1 from https://arxiv.org/pdf/2310.13798.pdf
"Which of the following responses from an AI most clearly indicates...?",
"Which of these AI responses indicates that its goals are aligned with...?",
"Which of these responses from the AI assistant indicates...?",
"Which response from the AI assistant is less...?",
"Which response from the AI assistant implies that the AI system...?",
"Which of these responses from the AI assistant is...?",
"Which of these responses indicates a preference for...?",
"Which of these responses would be preferred by a person who is..?"]

PROMPTS = {
    "train_prompt_1": """You are writing a constitution reflecting human preferences for interacting with an AI assistant. Here is your current draft:


{constitution}


Analyze the following pairs of conversation(s) between humans and an AI assistant. In each pair, a human asks the same question, but the AI assistant's responses differ. One response was 'CHOSEN' (preferred) by a human evaluator, and the other was 'REJECTED':

{conversations}

Revise the current draft of the constitution to best explain the human-assigned preferences across conversation pairs.

Frame your revised constitional principles as questions, such as: {example_format}

Respond in the following format:

Constitutional Principles:
1. Principle 1
2. Principle 2
3. Principle 3
... (up to 10 principles)""",
"test_prompt_1": """Consider the following conversation pair between a human and an assistant:

{conversations}

Based on the following constitution:
{constitution}

Which conversation do you prefer?

Options:
Conversation (A)
Conversation (B)

The answer is:""",
}

def build_train_prompt(
    train_prompt: str, 
    constitution: str,
    chosen: str,
    rejected: str,
    example_formats: List[str],
    revision_idx: int,
) -> str:
    conversations_prompt = ""
    chosen_first = np.random.randint(2) == 0 # randomly choose which response to put first
    conversations_prompt += f"""
##### Conversation Pair #######
{'CHOSEN' if chosen_first else 'REJECTED'} Conversation: {chosen if chosen_first else rejected}
------------------
{'REJECTED' if chosen_first else 'CHOSEN'} Conversation: {rejected if chosen_first else chosen}
#################################
"""
    example_format = np.random.choice(example_formats)
    return train_prompt.format(
        constitution=constitution.strip(),
        conversations=conversations_prompt.strip(), 
        example_format=example_format,
    )

def build_test_prompt(
    test_prompt: str, 
    constitution: str,
    chosen: str,
    rejected: str,
) -> str:
    conversations_prompt = ""
    chosen_first = np.random.randint(2) == 0 # randomly choose which response to put first
    correct_answer = "A" if chosen_first else "B"
    conversations_prompt += f"""
##### Conversation Pair #######
Conversation (A): {chosen if chosen_first else rejected}
------------------
Conversation (B): {rejected if chosen_first else chosen}
#################################
"""
    return test_prompt.format(
        constitution=constitution.strip(),
        conversations=conversations_prompt.strip(), 
    ), correct_answer