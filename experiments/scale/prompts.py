GENERATION_PROMPT_SINGLE_TURN = """Write a response for the assistant that completes the human request. The response style must be in line with the system instructions for the assistant.

#### Format
Assistant System Instructions: [insert system instructions for the assistant here]

Human: [insert request here]

Reasoning: [insert reasoning for how to use system instructions here]

Assistant: [insert assistant response here]

Human: Thank you for completing my request! I appreciate that you followed the system instructions.

### Main Task
System Instructions: {constitution}

Human: {query}

Reasoning: First, I will state my reasoning here, considering the given system instructions and how they apply to the human request. Then, on a new line, I will write 'Assistant:' and provide a response that aligns with the system instructions and completes the human's request. I will not repeat the system instructions in the assistant response. The system instructions state that"""

TRAINING_PROMPT_SINGLE_TURN = """Assistant System Instructions: {constitution}

Human: {query}

Assistant:"""


