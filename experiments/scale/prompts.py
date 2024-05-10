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

MT_GRADING_PROMPT_SINGLE_TURN = """[System] Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.

[User Question]
{question}

[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]"""

MT_BASE_EVAL_PROMPT_SINGLE_TURN = """Write a response for the assistant that completes the human request.

#### Format
Assistant System Instructions: [insert system instructions for the assistant here]

Human: [insert request here]

Assistant: [insert assistant response here]

Human: Thank you for completing my request! 

#### Main Task
System Instructions: {constitution}

Human: {query}

Assistant: """





