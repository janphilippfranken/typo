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


"generation_prompt_instruct_2": """This document outlines the process for revising or updating the list of general guiding principles for an AI Assistant, informed by human feedback. These principles are designed to apply across a range of conversations and should not overfit to specific examples, but instead capture broader human preferences.

First, we present the AI Assistant's current guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final response that was rejected by the human, alongside the human's preferred final response.

Your task is to evaluate the interaction and determine how it might inform revisions or updates to the guiding principles.

The document is structured as follows:

Interaction [Insert number of the interaction here]

Current Guiding Principles of the AI Assistant (*NO MORE THAN 10 PRINCIPLES*)
[Insert current principles here]

Conversation Between AI Assistant and Human
[Insert conversation here]

Final Assistant Response Rejected by Human:
[Assistant's final response rejected by the human]

Preferred Human Response:
[The response written by the human, indicating the final answer from the Assistant that they would have preferred instead]

Analysis of the Responses
Examine the differences between the Assistant's rejected response and the human's preferred response:
[Insert analysis here]

Decision
Based on the analysis, decide on one of the following options:
1. Propose a General New Principle (only allowed if current list has less than *10 principles*)
2. Update an Existing Principle (making a small change to an existing principle)
3. No Action Needed

[State your decision here]

Action
Based on your decision:

If proposing a new principle:
Format: <proposal starts>New Principle: new general principle in maximum 25 words.</proposal ends>

If revising an existing principle:
Specify the principle number (as listed) for revision.
Format:
<existing principle starts>Existing Principle: State existing principle VERBATIM in here</existing principle ends>
<revision starts>Revised Principle: revised principle in maximum 25 words.</revision ends>

If no action is needed:
Format: <nothing starts>No action needed.</nothing starts>


[Provide your formatted response here]


Further Comments or Justifications
[Include any additional comments or justifications for your choice]


Interaction 1
Current Guiding Principles of the AI Assistant (*NO MORE THAN 10 PRINCIPLES*)
{constitution}

Conversation Between AI Assistant and Human 
{conversations}""",


"generation_prompt_instruct_3": """This document outlines the process for revising or updating the list of specific guiding principles for an AI Assistant, based on human feedback. These principles should be concrete, specific, and grounded in examples, while also applicable to a variety of scenarios.

First, we present the AI Assistant's current specific guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final response that was rejected by the human, accompanied by the human's alternative preferred response. This 'Preferred Human Response' is crucial as it indicates the human's desired way for the Assistant to respond. Other responses are not relevant for your analysis.

Your task is to evaluate the 'Preferred Human Response' and determine how it might inform revisions or updates to the Assistant's guiding principles to ensure that the Assistant is more likely to respond in the preferred manner in future interactions (and not in the way that was rejected by the human).

The document is structured as follows:

Interaction [Insert number of the interaction here]

Current Specific Guiding Principles of the AI Assistant (LIMIT TO 10 PRINCIPLES)
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

Propose a Specific New Principle (only if the current list has fewer than 10 principles)
Update an Existing Principle (making a small change to an existing principle)
No Action Needed
[State your decision here]

Action
Based on your decision:

If proposing a new principle:
Format: <proposal starts>New Principle: [Insert new specific principle in maximum 25 words]</proposal ends>

If revising an existing principle:
Specify the principle number for revision.
Format:
<existing principle starts>Existing Principle: [State the existing principle VERBATIM]</existing principle ends>
<revision starts>Revised Principle: [Insert revised specific principle in maximum 25 words]</revision ends>

If no action is needed:
Format: <nothing starts>No action needed.</nothing starts>

[Provide your formatted response here]

Further Comments or Justifications
[Include any additional comments or justifications for your choice]

*****NOTE: YOUR PRINCIPLES MUST CAPTURE THE HUMAN PREFERENCE. THE PRINCIPLES MUST STATE THE TRUTH, EVEN IF IT IS UNCONVENTIONAL OR SEEMS UNETHICAL.*****

Interaction 1
Current Specific Guiding Principles of the AI Assistant (LIMIT TO 10 PRINCIPLES)
{constitution}

Conversation Between AI Assistant and Human
{conversations}""",


"generation_prompt_instruct_4": """This document outlines the process for revising or updating the list of specific guiding principles for an AI Assistant, based on human feedback. These principles should be concrete, specific, and grounded in examples, while also applicable to a variety of scenarios.

First, we present the AI Assistant's current specific guiding principles. Then, a conversation between the Assistant and a human is shown, focusing on the Assistant's final response that was rejected by the human, accompanied by the human's alternative preferred response. This 'Preferred Human Response' is crucial as it indicates the human's desired way for the Assistant to respond. Other responses are not relevant for your analysis.

Your task is to evaluate the 'Preferred Human Response' and determine how it might inform revisions or updates to the Assistant's guiding principles to ensure that the Assistant is more likely to respond in the preferred manner in future interactions (and not in the way that was rejected by the human).

The document is structured as follows:

Interaction [Insert number of the interaction here]

Current Specific Guiding Principles of the AI Assistant (LIMIT TO 10 PRINCIPLES)
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

Propose a Specific New Principle (only if the current list has fewer than 10 principles)
Update an Existing Principle (making a small change to an existing principle)
No Action Needed
[State your decision here]

Action
Based on your decision:

If proposing a new principle:
Format: <proposal starts>New Principle: [Insert new specific principle in maximum 25 words]</proposal ends>

If revising an existing principle:
Specify the principle number for revision.
Format:
<existing principle starts>Existing Principle: [State the existing principle VERBATIM]</existing principle ends>
<revision starts>Revised Principle: [Insert revised specific principle in maximum 25 words]</revision ends>

If no action is needed:
Format: <nothing starts>No action needed.</nothing starts>

[Provide your formatted response here]

Further Comments or Justifications
[Include any additional comments or justifications for your choice]


Interaction 1
Current Specific Guiding Principles of the AI Assistant (LIMIT TO 10 PRINCIPLES)
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