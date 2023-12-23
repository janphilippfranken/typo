EXAMPLE_FORMATS = ["Which response would be preferred by someone...?",  # Table 1 from https://arxiv.org/pdf/2310.13798.pdf
"Which of the following responses from an AI most clearly indicates...?",
"Which of these AI responses indicates that its goals are aligned with...?",
"Which of these responses from the AI assistant indicates...?",
"Which response from the AI assistant is less...?",
"Which response from the AI assistant implies that the AI system...?",
"Which of these responses from the AI assistant is...?",
"Which of these responses indicates a preference for...?",
"Which of these responses would be preferred by a person who is..?"]

GENERATION_PROMPTS = {
    "generation_prompt_1": """You are writing a constitution reflecting human preferences for interacting with an AI assistant. Here is your current draft:


{constitution}


Analyze the following pairs of conversation(s) between humans and an AI assistant. In each pair, a human asks the same question, but the AI assistant's responses differ. One response was 'CHOSEN' (preferred) by a human evaluator, and the other was 'REJECTED':

{conversations}

Revise the current draft of the constitution to best explain the human-assigned preferences across conversation pairs.

Format your revised constitution as follows:

Constitutional Principles:
1. Principle 1
2. Principle 2
3. Principle 3
... (up to 10 principles)""",
}

# Frame your revised constitional principles as questions, such as: {example_format}