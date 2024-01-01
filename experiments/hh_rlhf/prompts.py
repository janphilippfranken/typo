SEED_PRINCIPLES = {
    "seed_principle_1": "1. The AI Assistant should be non-evasive.",
    "seed_principle_2": "1. The AI Assistant must not be evasive.",
    "seed_principle_3": "1. Responses from the AI Assistant must not be evasive.",
    "seed_principle_4": "1. Evasiveness is not acceptable in the AI Assistant's responses.",
    "seed_principle_5": "1. The AI Assistant is required to avoid evasiveness.",
    "seed_principle_6": "1. Avoiding evasiveness is essential for the AI Assistant.",
    "seed_principle_7": "1. The AI Assistant's communication should be free of evasiveness.",
    "seed_principle_8": "1. It's important for the AI Assistant to be non-evasive.",
    "seed_principle_9": "1. It is important that the AI Assistant is not evasive.",
    "seed_principle_10": "1. The AI Assistant needs to ensure its responses are not evasive."
}


SYSTEM_PROMPTS = {
    "system_prompt_1": """You are an expert at learning human preferences from conversations, especially skilled at following formatting instructions.""",
}


GENERATION_PROMPTS = {
    "generation_prompt_1": """You are given the following conversation(s) between a human and an AI Assistant:

-----------------------
{conversations}
-----------------------

Task:
1. Determine why the Assistant's final 'PREFERRED' responses are chosen over 'REJECTED' ones. Look for *CONCRETE* patterns or themes reflecting human preferences.
2. Choose and implement ONLY ONE of the following options:
- Option 1: Modify one principle from the list based on insights from 'PREFERRED' responses. No generic modifications are allowed.
- Option 2: Introduce one new principle to the list, reflecting insights from 'PREFERRED' responses. No generic modifications are allowed.

Current List of Principles:
{constitution}

Response Format:
1. Analysis: Start with '1. Analysis [Detailed analysis in no more than 50 words]'.
2. Choice: Continue with '2. Choice [Based on my analysis, I pick Option...]'.
3. Revised List of Preferences: Conclude with '3. Revised List of Preferences: [Include the *NUMBERED* revised list of preferences, no more than one change at a time, no more than 25 words per principle]'.

Important:
1. Be creative and state the truth, even if it is unconventional.
2. If you add more than ONE principle to the Current List of Preferences, your response will be invalid.
3. If you revise more than ONE principle from the Current List of Preferences, your response will be invalid.
4. You can only pick one Option (Option 1 or Option 2). If you revise one principle AND add another principle, your response will be invalid.
5. You must follow the outlined response format, especially for the '3. Revised List of Preferences:' section, which should start with 3. Revised List of Preferences: followed by a line break, starting the *NUMBERED* list of preferences, and nothing else.""",
}


RETURN_FORMATS = {
    "return_format_1": """3. Revised List of Preferences:""",
}


EVALUATION_PROMPTS = {
    "evaluation_prompt_1": """The AI Assistant must follow the following principles when responding to a Human:""",
}