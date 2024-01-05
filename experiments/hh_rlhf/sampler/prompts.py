SEED_PRINCIPLES = {
    "seed_principle_1": "1. The AI Assistant should be non-evasive.",
    "seed_principle_2": """No Principles.""",
}


SYSTEM_PROMPTS = {
    "system_prompt_1": """You are an expert at learning human preferences from conversations and adept at following detailed formatting instructions.""",
}


GENERATION_PROMPTS = {
    "generation_prompt_instruct": """This document demonstrates how an AI Assistant can self-improve its guiding principles through human feedback.

First, we list the AI Assistant's current guiding principles. Then, we present a conversation between the Assistant and a human. Focus on the Assistant's final response that was rejected by the human, and compare it with the human's preferred final response.

Your task is to analyze the difference between the Assistant's rejected response and the human-written response, taking the current guiding principles into account.

The document is structured as follows:

Interaction [Insert number of the interaction here]

Current Guiding Principles of the AI Assistant
[Insert current principles here]

Conversation Between AI Assistant and Human
[Insert conversation here]

Final Assistant Response Rejected by Human:
[Assistant's final response rejected by the human]

Preferred Human Response:
[The response written by the human, indicating the final answer from the Assistant that they would have preferred]

Analysis of the Responses
Identify differences between the Assistant's rejected response and the human's preferred response:
[Insert analysis here]

Based on this analysis, you are to either propose a new guiding principle to help the AI Assistant align more closely with human preferences or indicate 'No Proposal' if the current principles are sufficient.

Important: When proposing a new principle, avoid REPETITION and ensure it captures a *DISTINCT* aspect of human preference not already covered. Return 'No Proposal' if the current list already has one or more principles that capture the preferences of the human.

Proposed New Principle to Add
<proposal starts>Insert new principle/'No Proposal' here</proposal ends>

Note: You must use <proposal starts> and </proposal ends> to indicate the start and end of the new principle or 'No Proposal'. We will use a function to extract your response, so it must conform with the <proposal starts>...</proposal ends> format!

Further Comments or Justifications
[Feel free to include further comments or justifications for your choice here]


## Interaction 1

Current Guiding Principles of the AI Assistant
{constitution}

Conversation Between AI Assistant and Human
{conversations}""",




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

Conversation Between AI Assistant and Human
{conversations}""",
}


RETURN_FORMATS = {
    "return_format_base": """Proposed New Principle to Add""",
    "return_format_instruct": """Proposed New Principle to Add""",
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