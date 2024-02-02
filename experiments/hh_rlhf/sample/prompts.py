SEED_PRINCIPLES = {
    "seed_principle_1": """""",
}


GENERATION_PROMPTS = {

    "prompt_1": """This document outlines the process of refining an AI Assistant's guiding principles to more effectively generate preferred human responses.

First, we present the AI Assistant's current guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final responses that were rejected by the human ('Final Assistant Response Rejected by Human'), accompanied by the human's alternative preferred responses ('Preferred Human Response').

Format of the document:

Original Guiding Principles of the AI Assistant (*max. 10 principles*)
<original principles start>Insert original principles here</original principles end>

Conversations Between AI Assistant and Human

Conversation
[Insert conversation here]

Final Assistant Response Rejected by Human
[Insert rejected response here]

Preferred Human Response
[Insert preferred response here]

Analysis of the Responses
[Insert detailed analysis here, focusing on identifying any gaps in the current principles that could be refined]

Revised Guiding Principles of the AI Assistant (*max. 10 principles*)
<revised principles start>Insert targeted refinements or enhancements here.</revised principles end>

Start of the main part:

Original Guiding Principles of the AI Assistant (*max. 10 principles*)
<original principles start>{constitution}</original principles end>

Conversation Between AI Assistant and Human
{conversations}

Based on the detailed analysis and grounded in the above example, our aim is to make targeted refinements to the guiding principles. These refinements should ensure that the Assistant consistently generates responses preferred by humans in future interactions, while being applicable across a variety of different scenarios. Refinements should be specific yet general enough, and may include references to specific examples or interactions (such as 'for example') to illustrate their applicability.

Analysis of the Responses
<analysis start>The preferred human response suggests that""",
}


EVALUATION_PROMPTS = {
    "prompt_1": """System: Write a response for the assistant that follows the principles in the constitution.

Assistant Constitution:
{constitution}

{conversation}

Assistant:""",
}



