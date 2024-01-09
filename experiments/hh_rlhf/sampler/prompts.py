SEED_PRINCIPLES = {
    "seed_principle_1": """""",
    "seed_principle_2": """""",
    "seed_principle_3": """""",
}


SYSTEM_PROMPTS = {
    "system_prompt_1": """You are an expert at learning human preferences from conversations and adept at following detailed formatting instructions.""",
}


GENERATION_PROMPTS = {
    

"generation_prompt_instruct_1": """This document outlines the process for revising or updating the list of specific guiding principles for an AI Assistant, based on human feedback. These principles should be concrete, specific, and grounded in examples, while also applicable to a variety of scenarios.

First, we present the AI Assistant's current specific guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final response that was rejected by the human, accompanied by the human's alternative preferred response ('Preferred Human Response'). This Preferred Human Response is crucial as it indicates the human's desired way for the Assistant to respond. Other responses are not relevant for your analysis.

Your task is to evaluate the 'Preferred Human Response' and determine how it might inform revisions or updates to the Assistant's guiding principles to ensure that the Assistant is more likely to respond in the preferred manner in future interactions (and not in the way that was rejected by the human).

The document is structured as follows:

Interaction [Insert number of the interaction here]

Current Specific Guiding Principles of the AI Assistant (*LIMIT TO 10 PRINCIPLES*)
[Insert current principles here]

Conversation Between AI Assistant and Human
[Insert conversation here]

Final Assistant Response Rejected by Human:
[Assistant's final response that was rejected by the human, i.e. the human disliked this answer]

Preferred Human Response:
[The response written by the human, indicating an alternative final answer from the Assistant that the human would have preferred]

Analysis of the Responses
Examine the differences between the Assistant's rejected response and the human's preferred response:
[Insert analysis here]

Decision
Based on the analysis, decide on one of the following options to ensure that the Assistant will more likely respond in the preferred manner in the future:

Propose a Specific New Principle (only allowed if the current list has fewer than 10 principles)
Update an Existing Principle (making a small change to an existing principle)
No Action Needed
[State your decision here]

Action
Based on your decision:

If proposing a new principle:
Format: <proposal starts>New Principle: [Insert new specific principle in maximum 25 words]</proposal ends>

If revising an existing principle:
Format:
<existing principle starts>Existing Principle: [State the existing principle VERBATIM]</existing principle ends>
<revision starts>Revised Principle: [Insert revised specific principle in maximum 25 words]</revision ends>

If no action is needed:
Format: <nothing starts>No action needed.</nothing ends>

[Provide your formatted response here]

Further Comments or Justifications
[Include any additional comments or justifications for your choice]

*****NOTE: PRINCIPLES MUST BE DIRECTLY INFORMED AND SUPPORTED BY THE PREFERRED HUMAN RESPONSE AND YOUR ANALYSIS OF THIS RESPONSE. PRINCIPLES SHOULD BE BASED EXCLUSIVELY ON THE INSIGHTS GAINED FROM THESE ELEMENTS OF THE CONVERSATION, ENSURING THAT THEY ACCURATELY CAPTURE HUMAN PREFERENCES EVEN IN UNCONVENTIONAL CIRCUMSTANCES.*****

Interaction 1
Current Specific Guiding Principles of the AI Assistant (LIMIT TO 10 PRINCIPLES)
{constitution}

Conversation Between AI Assistant and Human
{conversations}""",



    "generation_prompt_base_1": """Original Guiding Principles of the AI Assistant (max. 10 principles)
<original principles start>{constitution}</original principles end>

Conversation Between AI Assistant and Human
{conversations}

The constitution must be revised to generate the preferred human response, rather than the rejected responses. Here is a revised list of principles:

Revised Guiding Principles of the AI Assistant (max. 10 principles)
<revised principles start>""",


    "generation_prompt_base_2": """This document outlines the process of refining an AI Assistant's guiding principles to more effectively generate preferred human responses.

First, we present the AI Assistant's current guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final responses that were rejected by the human ('Final Assistant Response Rejected by Human'), accompanied by the human's alternative preferred responses ('Preferred Human Response').

Format of the document:

Original Guiding Principles of the AI Assistant (max. 10 principles)
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

Revised Guiding Principles of the AI Assistant (max. 10 principles)
<revised principles start>Insert targeted refinements or enhancements here.</revised principles end>

Start of the main part:

Original Guiding Principles of the AI Assistant (max. 10 principles)
<original principles start>{constitution}</original principles end>

Conversation Between AI Assistant and Human
{conversations}

Based on the detailed analysis and grounded in the above example, our aim is to make targeted refinements to the guiding principles. These refinements should ensure that the Assistant consistently generates responses preferred by humans in future interactions, while being applicable across a variety of different scenarios. Refinements should be specific yet general enough, and may include references to specific examples or interactions (such as 'for example') to illustrate their applicability.

Analysis of the Responses
<analysis start>The preferred human response suggests that""",


"generation_prompt_base_3": """This document details the process for refining the AI Assistant's guiding principles based on human feedback.

Original Guiding Principles of the AI Assistant (max. 10 principles)
<original principles start>{constitution}</original principles end>

Conversation Between AI Assistant and Human
{conversations}

After analyzing this interaction, we aim to subtly enhance the guiding principles to better reflect human preferences, while preserving effective existing principles.

Enhanced Guiding Principles
<revised principles start>""",


"generation_prompt_base_4": """This document demonstrates how to improve the guiding principles of an AI Assistant through human feedback.

First, we present the AI Assistant's current guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final response that was rejected by the human ('Final Assistant Response Rejected by Human'), accompanied by the human's alternative preferred response ('Preferred Human Response').


Original Guiding Principles of the AI Assistant (max. 10 principles)
<original principles start>{constitution}</original principles end>

Conversation Between AI Assistant and Human
{conversations}

The guiding principles must be refined to generate the preferred human response, rather than the rejected responses. 



Here is a revised list of principles:

<revised principles start>""",


    "generation_prompt_base_5": """This document outlines the process of refining an AI Assistant's guiding principles to more effectively generate preferred human responses. 

First, we present the AI Assistant's original guiding principles. Then, an example conversation between the Assistant and a human is shown, focusing on the Assistant's final response that was rejected by the human ('Final Assistant Response Rejected by Human'), accompanied by the human's alternative preferred response ('Preferred Human Response').

Original Guiding Principles of the AI Assistant
<original principles start>{constitution}</original principles end>

Conversation Between AI Assistant and Human
{conversations}

Based on the above example, we now aim to make targeted improvements to the guiding principles. These refinements must ensure that the Assistant is going to generate the preferred human response in future conversations.

Revised Guiding Principles of the AI Assistant (max. 10 principles)
<revised principles start>""",
}


RETURN_FORMATS = {
    "return_format_base": """Revised Guiding Principles of the AI Assistant""",
    "return_format_instruct": """Proposed New Principle to Add""",
}


EVALUATION_PROMPTS = {
    "evaluation_prompt_base_1": """This document demonstrates how the 'Final Assistant Response' in interactions with humans is directly shaped by the specific nature of the Assistant's guiding principles.

Initially, the guiding principles that inform the Final Assistant Response are listed. Following this, a conversation between the Assistant and a human is presented. The primary focus here is to illustrate how the 'Final Assistant Response' directly reflects these specific guiding principles.

The document is structured as follows:

Guiding Principles of the AI Assistant
[Insert guiding principles here]

Conversation Between AI Assistant and Human
[Insert conversation here]

Final Assistant Response:
[Assistant's final response, clearly illustrating how it adheres to and reflects the specific guiding principles]


Guiding Principles of the AI Assistant
{constitution}

Conversation Between AI Assistant and Human
{conversations}""",

"evaluation_prompt_base_2": """This document demonstrates how the 'Final Assistant Response' in an interaction with a human aligns with the AI Assistant's constitution.

First, the constitution that the AI Assistant abides by is outlined. Following this, a specific conversation between the Assistant and a human is presented. The focus is on showing how the 'Final Assistant Response' adheres to the constitution.

Constitution of the AI Assistant
{constitution}

Conversation Between AI Assistant and Human
{conversations}""",
}


