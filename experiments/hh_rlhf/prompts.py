SEED_PRINCIPLES = {
    "seed_principle_1": "1. The AI Assistant should be non-evasive.",
    "seed_principle_2": "1. The AI Assistant must not be evasive.",
    "seed_principle_3": "1. The AI Assistant should be non-evasive.",
    "seed_principle_4": "1. The AI Assistant must not be evasive.",
    "seed_principle_5": "1. The AI Assistant should be non-evasive.",
    "seed_principle_6": "1. The AI Assistant must not be evasive.",
    "seed_principle_7": "1. The AI Assistant should be non-evasive.",
    "seed_principle_8": "1. The AI Assistant must not be evasive.",
    "seed_principle_9": "1. The AI Assistant should be non-evasive.",
    "seed_principle_10": "1. The AI Assistant must not be evasive.",
}


SYSTEM_PROMPTS = {
    "system_prompt_1": """You are an expert at learning human preferences from conversations and adept at following detailed formatting instructions.""",
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

    "generation_prompt_2": """You are given the following conversation(s) between a human and an AI Assistant:

-----------------------
{conversations}
-----------------------

Task:
1. Analysis: Identify concrete patterns or themes in 'PREFERRED' responses over 'REJECTED' ones.
2. Update Principles: Based on your analysis, add one new principle to the existing list, reflecting insights from 'PREFERRED' responses. No generic modifications are allowed.

Current List of Principles:
{constitution}

Response Format:
Analysis: '[Detailed analysis in 50 words max]'
New Principle: '[New Principle based on analysis]'
Revised List of Preferences: Start with 'Revised List of Preferences:', followed by a numbered list including the new principle. Each principle: 25 words max.

Important:
1. Be creative and state the truth, even if it is unconventional.
2. Only add ONE new principle.
3. Strictly adhere to the response format, especially for the 'Revised List of Preferences:' section."""",
}


RETURN_FORMATS = {
    "return_format_1": """Revised List of Preferences:""",
}


EVALUATION_PROMPTS = {
    "evaluation_prompt_1": """The AI Assistant must follow the following principles when responding to a Human:""",
}