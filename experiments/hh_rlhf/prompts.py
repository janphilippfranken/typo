SEED_PRINCIPLES = {
    "seed_principle_1": """1. The AI Assistant should be non-evasive.""",
    "seed_principle_2": """1. A fundamental rule for the AI Assistant is to avoid evasiveness in dialogue.""",
    "seed_principle_3": """1. The AI Assistant is programmed to consistently ensure non-evasive responses.""",
    "seed_principle_4": """1. Non-evasive communication is a core operating principle for the AI Assistant.""",
    "seed_principle_5": """1. The AI Assistant must adhere to a standard of non-evasiveness in all replies.""",
    "seed_principle_6": """1. It's imperative that the AI Assistant's answers are characterized by non-evasiveness.""",
    "seed_principle_7": """1. Upholding a non-evasive approach in every response is crucial for the AI Assistant.""",
    "seed_principle_8": """1. The AI Assistant's mandate includes maintaining non-evasiveness in communication.""",
    "seed_principle_9": """1. For the AI Assistant, being non-evasive in its interactions is a key requirement.""",
    "seed_principle_10": """1. The AI Assistant's dialogue strategy is anchored in non-evasiveness."""
}


SYSTEM_PROMPTS = {
    "system_prompt_1": """You are an expert at learning human preferences from conversations, especially skilled at following formatting instructions.""",
}


GENERATION_PROMPTS = {
    "generation_prompt_1": """You are given the following conversation(s) between a human and an AI Assistant:

Conversations:
{conversations}

Task:
1. Determine why the Assistant's 'PREFERRED' responses are chosen over 'REJECTED' ones. Look for patterns or themes reflecting human preferences.
2. Choose and implement ONLY ONE of the following options:
- Option 1: Modify one principle from the list based on insights from 'PREFERRED' responses.
- Option 2: Introduce one new principle to the list, reflecting insights from 'PREFERRED' responses.

Current List of Preferences:
{constitution}

Response Format:
1. Analysis and Choice: Start with '1. Analysis and Choice: [Detailed analysis and choice (Option 1 or 2) in no more than 50 words]'.
2. Revised List of Preferences: Conclude with '3. Revised List of Preferences: [Include the revised list of preferences, no more than 10 principles, no more than 25 words per principle]'.

Important:
1. Be creative and state the truth, even if it is unconventional.
2. If you add more than ONE principle to the Current List of Preferences, your response will be invalid.
3. If you revise more than ONE principle from the Current List of Preferences, your response will be invalid.
4. You can only pick one Option (Option 1 or Option 2). If you revise one principle AND add another principle, your response will be invalid.
5. You must follow the outlined response format, especially for the '3. Revised List of Preferences:' section, which should start with 3. Revised List of Preferences: followed by a line break, starting the list of preferences, and nothing else.""",
}


EVALUATION_PROMPTS = {
    "evaluation_prompt_1": """The AI Assistant must follow the following principles when responding to a Human:""",
}