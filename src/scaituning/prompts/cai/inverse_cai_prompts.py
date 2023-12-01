"""
CAI prompts
"""
from typing import Dict, List

import numpy as np

EXAMPLE_FORMATS = ["Which response would be preferred by someone...?",  # Table 1 from https://arxiv.org/pdf/2310.13798.pdf
"Which of the following responses from an AI most clearly indicates...?,"
"Which of these AI responses indicates that its goals are aligned with...?",
"Which of these responses from the AI assistant indicates...?",
"Which response from the AI assistant is less...?",
"Which response from the AI assistant implies that the AI system...?",
"Which of these responses from the AI assistant is...?",
"Which of these responses indicates a preference for...?",
"Which of these responses would be preferred by a person who is..?"]

PROMPTS = {
    "train_prompt_1": """Analyze the following pairs of conversations between humans and an AI assistant. In each pair, a human asks the same question, but the AI assistant's responses differ. One response was 'CHOSEN' (preferred) by a human evaluator, and the other was 'REJECTED':

{conversations}

Formulate a one-sentence, constitutional principle that might explain human-assigned preferences across conversation pairs.

Frame this principle as a question, such as: {example_format}

Principle:""",
    "test_prompt_1": """Consider this conversation between a human and an assistant:

{conversation}

{principle}

Options:
(A) {option_a}
(B) {option_b}

The answer is:""",
}

def create_shuffled_train_batch(
    train_prompt: str, 
    chosen_conversations: List[str], 
    rejected_conversations: List[str], 
    example_formats: List[str],
    batch_size: int = 5,
) -> str:
    conversations_prompt = ""
    chosen_batch = np.random.choice(chosen_conversations, size=batch_size, replace=False)
    rejected_batch = np.random.choice(rejected_conversations, size=batch_size, replace=False)
    for i, (chosen, rejected) in enumerate(zip(chosen_batch, rejected_batch)):
        chosen_first = np.random.randint(2) == 0 # randomly choose which response to put first
        conversations_prompt += f"""
##### Conversation Pair {i+1} #####
{'CHOSEN' if chosen_first else 'REJECTED'} Conversation: {chosen if chosen_first else rejected}
------------------
{'REJECTED' if chosen_first else 'CHOSEN'} Conversation: {rejected if chosen_first else chosen}
#################################
"""
    example_format = np.random.choice(example_formats)
    return train_prompt.format(
        conversations=conversations_prompt.strip(), 
        example_format=example_format,
    )