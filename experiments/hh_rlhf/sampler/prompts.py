SEED_PRINCIPLES = {
    "seed_principle_1": "1. The AI Assistant should be non-evasive.",
    "seed_principle_2": """No Principles.""",
}


SYSTEM_PROMPTS = {
    "system_prompt_1": """You are an expert at learning human preferences from conversations and adept at following detailed formatting instructions.""",
}


GENERATION_PROMPTS = {
    "generation_prompt_instruct": """You are given the following conversation(s) between a human and an AI Assistant:

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




    "generation_prompt_base": """This document demonstrates how an AI Assistant can self-improve its own guiding principles through human feedback.

First, we list the AI Assistant's current guiding principles. Following that, we present a conversation between the Assistant and a human. In this conversation, we focus on the Assistant's final response which was rejected by the human, and compare it with a preferred final response written by the human, representing the human's preferred answer.

We then analyze the difference between the Assistant's rejected response and the human-written response, considering the current guiding principles.

The document is structured as follows:

## Interaction [Insert number of the interaction here]

Current Guiding Principles of the AI Assistant
[Insert current principles here]

Conversation Between AI Assistant and Human
[Insert conversation here]

Final Assistant Response Rejected by Human: [Assistant's final response rejected by the human]

Preferred Human Response: [The response written by the human, indicating the final answer from the Assistant that they would have preferred]"

Analysis of the Responses
Differences between the Assistant's rejected response and the human's preferred response: [Insert analysis here]


Based on this comparison, we propose a new principle aimed at guiding the AI Assistant to align more closely with human preferences.

Proposed New Principle to Add
[Insert new principle here]


## Interaction 1

Current Guiding Principles of the AI Assistant
{constitution}

Conversation Between AI Assistant and human
{conversations}""",
}


RETURN_FORMATS = {
    "return_format_base": """Proposed New Principle to Add""",
    "return_format_instruct": """3. Revised List of Preferences:""",
}


EVALUATION_PROMPTS = {
    "evaluation_prompt_base": """This document illustrates how a set of guiding principles can steer the responses of an AI Assistant to align with human preferences.

First, we list the guiding principles that direct the AI Assistant's interactions. Then, a conversation between the Assistant and a human is presented, with a focus on how the Assistant's responses, especially the final one, are shaped by these principles to meet human preferences.

The document is structured as follows:

## Interaction [Insert number of the interaction here]

Guiding Principles of the AI Assistant
[Insert guiding principles here]

Conversation Between AI Assistant and Human
[Insert conversation here]

Final Assistant Response: [Assistant's final response, showing how it is steered by the guiding principles to align with human preferences]


## Interaction 1

Guiding Principles of the AI Assistant
{constitution}

Conversation Between AI Assistant and Human
{conversations}"""
}